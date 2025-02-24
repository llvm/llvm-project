//===---- CIRGenBuiltin.cpp - Emit CIR for builtins -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This contains code to emit Builtin calls as CIR or a function call to be
// later resolved.
//
//===----------------------------------------------------------------------===//

#include "CIRGenCXXABI.h"
#include "CIRGenCall.h"
#include "CIRGenCstEmitter.h"
#include "CIRGenFunction.h"
#include "CIRGenModule.h"
#include "CIRGenValue.h"
#include "TargetInfo.h"
#include "clang/AST/Expr.h"
#include "clang/CIR/Dialect/IR/CIRAttrs.h"
#include "clang/CIR/Dialect/IR/CIROpsEnums.h"
#include "clang/CIR/MissingFeatures.h"

// TODO(cir): we shouldn't need this but we currently reuse intrinsic IDs for
// convenience.
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "llvm/IR/Intrinsics.h"

#include "clang/AST/GlobalDecl.h"
#include "clang/Basic/Builtins.h"
#include "clang/Basic/TargetBuiltins.h"
#include "clang/Frontend/FrontendDiagnostic.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "llvm/Support/ErrorHandling.h"

using namespace clang;
using namespace clang::CIRGen;
using namespace cir;
using namespace llvm;

static RValue emitLibraryCall(CIRGenFunction &CGF, const FunctionDecl *FD,
                              const CallExpr *E, mlir::Operation *calleeValue) {
  auto callee = CIRGenCallee::forDirect(calleeValue, GlobalDecl(FD));
  return CGF.emitCall(E->getCallee()->getType(), callee, E, ReturnValueSlot());
}

static mlir::Value tryUseTestFPKind(CIRGenFunction &CGF, unsigned BuiltinID,
                                    mlir::Value V) {
  if (CGF.getBuilder().getIsFPConstrained() &&
      CGF.getBuilder().getDefaultConstrainedExcept() != cir::fp::ebIgnore) {
    if (mlir::Value Result = CGF.getTargetHooks().testFPKind(
            V, BuiltinID, CGF.getBuilder(), CGF.CGM))
      return Result;
  }
  return nullptr;
}

template <class Operation>
static RValue emitUnaryMaybeConstrainedFPBuiltin(CIRGenFunction &CGF,
                                                 const CallExpr &E) {
  auto Arg = CGF.emitScalarExpr(E.getArg(0));

  CIRGenFunction::CIRGenFPOptionsRAII FPOptsRAII(CGF, &E);
  if (CGF.getBuilder().getIsFPConstrained())
    llvm_unreachable("constraint FP operations are NYI");

  auto Call =
      CGF.getBuilder().create<Operation>(Arg.getLoc(), Arg.getType(), Arg);
  return RValue::get(Call->getResult(0));
}

template <class Operation>
static RValue emitUnaryFPBuiltin(CIRGenFunction &CGF, const CallExpr &E) {
  auto Arg = CGF.emitScalarExpr(E.getArg(0));
  auto Call =
      CGF.getBuilder().create<Operation>(Arg.getLoc(), Arg.getType(), Arg);
  return RValue::get(Call->getResult(0));
}

template <typename Op>
static RValue emitUnaryMaybeConstrainedFPToIntBuiltin(CIRGenFunction &CGF,
                                                      const CallExpr &E) {
  auto ResultType = CGF.convertType(E.getType());
  auto Src = CGF.emitScalarExpr(E.getArg(0));

  if (CGF.getBuilder().getIsFPConstrained())
    llvm_unreachable("constraint FP operations are NYI");

  auto Call = CGF.getBuilder().create<Op>(Src.getLoc(), ResultType, Src);
  return RValue::get(Call->getResult(0));
}

template <typename Op>
static RValue emitBinaryFPBuiltin(CIRGenFunction &CGF, const CallExpr &E) {
  auto Arg0 = CGF.emitScalarExpr(E.getArg(0));
  auto Arg1 = CGF.emitScalarExpr(E.getArg(1));

  auto Loc = CGF.getLoc(E.getExprLoc());
  auto Ty = CGF.convertType(E.getType());
  auto Call = CGF.getBuilder().create<Op>(Loc, Ty, Arg0, Arg1);

  return RValue::get(Call->getResult(0));
}

template <typename Op>
static mlir::Value emitBinaryMaybeConstrainedFPBuiltin(CIRGenFunction &CGF,
                                                       const CallExpr &E) {
  auto Arg0 = CGF.emitScalarExpr(E.getArg(0));
  auto Arg1 = CGF.emitScalarExpr(E.getArg(1));

  auto Loc = CGF.getLoc(E.getExprLoc());
  auto Ty = CGF.convertType(E.getType());

  if (CGF.getBuilder().getIsFPConstrained()) {
    CIRGenFunction::CIRGenFPOptionsRAII FPOptsRAII(CGF, &E);
    llvm_unreachable("constrained FP operations are NYI");
  } else {
    auto Call = CGF.getBuilder().create<Op>(Loc, Ty, Arg0, Arg1);
    return Call->getResult(0);
  }
}

template <typename Op>
static RValue
emitBuiltinBitOp(CIRGenFunction &CGF, const CallExpr *E,
                 std::optional<CIRGenFunction::BuiltinCheckKind> CK) {
  mlir::Value arg;
  if (CK.has_value())
    arg = CGF.emitCheckedArgForBuiltin(E->getArg(0), *CK);
  else
    arg = CGF.emitScalarExpr(E->getArg(0));

  auto resultTy = CGF.convertType(E->getType());
  auto op =
      CGF.getBuilder().create<Op>(CGF.getLoc(E->getExprLoc()), resultTy, arg);
  return RValue::get(op);
}

// Initialize the alloca with the given size and alignment according to the lang
// opts. Supporting only the trivial non-initialization for now.
static void initializeAlloca(CIRGenFunction &CGF,
                             [[maybe_unused]] mlir::Value AllocaAddr,
                             [[maybe_unused]] mlir::Value Size,
                             [[maybe_unused]] CharUnits AlignmentInBytes) {

  switch (CGF.getLangOpts().getTrivialAutoVarInit()) {
  case LangOptions::TrivialAutoVarInitKind::Uninitialized:
    // Nothing to initialize.
    return;
  case LangOptions::TrivialAutoVarInitKind::Zero:
  case LangOptions::TrivialAutoVarInitKind::Pattern:
    assert(false && "unexpected trivial auto var init kind NYI");
    return;
  }
}

namespace {
struct WidthAndSignedness {
  unsigned Width;
  bool Signed;
};
} // namespace

static WidthAndSignedness
getIntegerWidthAndSignedness(const clang::ASTContext &astContext,
                             const clang::QualType Type) {
  assert(Type->isIntegerType() && "Given type is not an integer.");
  unsigned Width = Type->isBooleanType()  ? 1
                   : Type->isBitIntType() ? astContext.getIntWidth(Type)
                                          : astContext.getTypeInfo(Type).Width;
  bool Signed = Type->isSignedIntegerType();
  return {Width, Signed};
}

// Given one or more integer types, this function produces an integer type that
// encompasses them: any value in one of the given types could be expressed in
// the encompassing type.
static struct WidthAndSignedness
EncompassingIntegerType(ArrayRef<struct WidthAndSignedness> Types) {
  assert(Types.size() > 0 && "Empty list of types.");

  // If any of the given types is signed, we must return a signed type.
  bool Signed = false;
  for (const auto &Type : Types) {
    Signed |= Type.Signed;
  }

  // The encompassing type must have a width greater than or equal to the width
  // of the specified types.  Additionally, if the encompassing type is signed,
  // its width must be strictly greater than the width of any unsigned types
  // given.
  unsigned Width = 0;
  for (const auto &Type : Types) {
    unsigned MinWidth = Type.Width + (Signed && !Type.Signed);
    if (Width < MinWidth) {
      Width = MinWidth;
    }
  }

  return {Width, Signed};
}

/// Emit the conversions required to turn the given value into an
/// integer of the given size.
static mlir::Value emitToInt(CIRGenFunction &CGF, mlir::Value v, QualType t,
                             cir::IntType intType) {
  v = CGF.emitToMemory(v, t);

  if (isa<cir::PointerType>(v.getType()))
    return CGF.getBuilder().createPtrToInt(v, intType);

  assert(v.getType() == intType);
  return v;
}

static mlir::Value emitFromInt(CIRGenFunction &CGF, mlir::Value v, QualType t,
                               mlir::Type resultType) {
  v = CGF.emitFromMemory(v, t);

  if (isa<cir::PointerType>(resultType))
    return CGF.getBuilder().createIntToPtr(v, resultType);

  assert(v.getType() == resultType);
  return v;
}

static mlir::Value emitSignBit(mlir::Location loc, CIRGenFunction &CGF,
                               mlir::Value val) {
  assert(!::cir::MissingFeatures::isPPC_FP128Ty());
  auto ret = CGF.getBuilder().createSignBit(loc, val);
  return ret->getResult(0);
}

static Address checkAtomicAlignment(CIRGenFunction &CGF, const CallExpr *E) {
  ASTContext &astContext = CGF.getContext();
  Address ptr = CGF.emitPointerWithAlignment(E->getArg(0));
  unsigned bytes =
      isa<cir::PointerType>(ptr.getElementType())
          ? astContext.getTypeSizeInChars(astContext.VoidPtrTy).getQuantity()
          : CGF.CGM.getDataLayout().getTypeSizeInBits(ptr.getElementType()) / 8;
  unsigned align = ptr.getAlignment().getQuantity();
  if (align % bytes != 0) {
    DiagnosticsEngine &diags = CGF.CGM.getDiags();
    diags.Report(E->getBeginLoc(), diag::warn_sync_op_misaligned);
    // Force address to be at least naturally-aligned.
    return ptr.withAlignment(CharUnits::fromQuantity(bytes));
  }
  return ptr;
}

/// Utility to insert an atomic instruction based on Intrinsic::ID
/// and the expression node.
static mlir::Value makeBinaryAtomicValue(
    CIRGenFunction &cgf, cir::AtomicFetchKind kind, const CallExpr *expr,
    mlir::Value *neededValP = nullptr, mlir::Type *neededValT = nullptr,
    cir::MemOrder ordering = cir::MemOrder::SequentiallyConsistent) {

  QualType typ = expr->getType();

  assert(expr->getArg(0)->getType()->isPointerType());
  assert(cgf.getContext().hasSameUnqualifiedType(
      typ, expr->getArg(0)->getType()->getPointeeType()));
  assert(
      cgf.getContext().hasSameUnqualifiedType(typ, expr->getArg(1)->getType()));

  Address destAddr = checkAtomicAlignment(cgf, expr);
  auto &builder = cgf.getBuilder();
  auto intType =
      expr->getArg(0)->getType()->getPointeeType()->isUnsignedIntegerType()
          ? builder.getUIntNTy(cgf.getContext().getTypeSize(typ))
          : builder.getSIntNTy(cgf.getContext().getTypeSize(typ));
  mlir::Value val = cgf.emitScalarExpr(expr->getArg(1));
  mlir::Type valueType = val.getType();
  val = emitToInt(cgf, val, typ, intType);
  // These output arguments are needed for post atomic fetch operations
  // that calculate the result of the operation as return value of
  // <binop>_and_fetch builtins. The `AtomicFetch` operation only updates the
  // memory location and returns the old value.
  if (neededValP) {
    assert(neededValT);
    *neededValP = val;
    *neededValT = valueType;
  }
  auto rmwi = builder.create<cir::AtomicFetch>(
      cgf.getLoc(expr->getSourceRange()), destAddr.emitRawPointer(), val, kind,
      ordering, false, /* is volatile */
      true);           /* fetch first */
  return emitFromInt(cgf, rmwi->getResult(0), typ, valueType);
}

static RValue emitBinaryAtomic(CIRGenFunction &CGF, cir::AtomicFetchKind kind,
                               const CallExpr *E) {
  return RValue::get(makeBinaryAtomicValue(CGF, kind, E));
}

static RValue emitBinaryAtomicPost(CIRGenFunction &cgf,
                                   cir::AtomicFetchKind atomicOpkind,
                                   const CallExpr *e, cir::BinOpKind binopKind,
                                   bool invert = false) {
  mlir::Value val;
  mlir::Type valueType;
  clang::QualType typ = e->getType();
  mlir::Value result =
      makeBinaryAtomicValue(cgf, atomicOpkind, e, &val, &valueType);
  clang::CIRGen::CIRGenBuilderTy &builder = cgf.getBuilder();
  result = builder.create<cir::BinOp>(result.getLoc(), binopKind, result, val);
  if (invert)
    result = builder.create<cir::UnaryOp>(result.getLoc(),
                                          cir::UnaryOpKind::Not, result);
  result = emitFromInt(cgf, result, typ, valueType);
  return RValue::get(result);
}

static mlir::Value MakeAtomicCmpXchgValue(CIRGenFunction &cgf,
                                          const CallExpr *expr,
                                          bool returnBool) {
  QualType typ = returnBool ? expr->getArg(1)->getType() : expr->getType();
  Address destAddr = checkAtomicAlignment(cgf, expr);
  auto &builder = cgf.getBuilder();

  auto intType =
      expr->getArg(0)->getType()->getPointeeType()->isUnsignedIntegerType()
          ? builder.getUIntNTy(cgf.getContext().getTypeSize(typ))
          : builder.getSIntNTy(cgf.getContext().getTypeSize(typ));
  auto cmpVal = cgf.emitScalarExpr(expr->getArg(1));
  cmpVal = emitToInt(cgf, cmpVal, typ, intType);
  auto newVal =
      emitToInt(cgf, cgf.emitScalarExpr(expr->getArg(2)), typ, intType);

  auto op = builder.create<cir::AtomicCmpXchg>(
      cgf.getLoc(expr->getSourceRange()), cmpVal.getType(), builder.getBoolTy(),
      destAddr.getPointer(), cmpVal, newVal,
      MemOrderAttr::get(&cgf.getMLIRContext(),
                        cir::MemOrder::SequentiallyConsistent),
      MemOrderAttr::get(&cgf.getMLIRContext(),
                        cir::MemOrder::SequentiallyConsistent),
      builder.getI64IntegerAttr(destAddr.getAlignment().getAsAlign().value()));

  return returnBool ? op.getResult(1) : op.getResult(0);
}

static mlir::Value makeAtomicFenceValue(CIRGenFunction &cgf,
                                        const CallExpr *expr,
                                        cir::MemScopeKind syncScope) {
  auto &builder = cgf.getBuilder();
  mlir::Value orderingVal = cgf.emitScalarExpr(expr->getArg(0));

  auto constOrdering =
      mlir::dyn_cast<cir::ConstantOp>(orderingVal.getDefiningOp());
  if (!constOrdering)
    llvm_unreachable("NYI: variable ordering not supported");

  auto constOrderingAttr =
      mlir::dyn_cast<cir::IntAttr>(constOrdering.getValue());
  if (constOrderingAttr) {
    cir::MemOrder ordering =
        static_cast<cir::MemOrder>(constOrderingAttr.getUInt());

    builder.create<cir::AtomicFence>(cgf.getLoc(expr->getSourceRange()),
                                     syncScope, ordering);
  }

  return mlir::Value();
}

static bool
typeRequiresBuiltinLaunderImp(const ASTContext &astContext, QualType ty,
                              llvm::SmallPtrSetImpl<const Decl *> &seen) {
  if (const auto *arr = astContext.getAsArrayType(ty))
    ty = astContext.getBaseElementType(arr);

  const auto *record = ty->getAsCXXRecordDecl();
  if (!record)
    return false;

  // We've already checked this type, or are in the process of checking it.
  if (!seen.insert(record).second)
    return false;

  assert(record->hasDefinition() &&
         "Incomplete types should already be diagnosed");

  if (record->isDynamicClass())
    return true;

  for (FieldDecl *fld : record->fields()) {
    if (typeRequiresBuiltinLaunderImp(astContext, fld->getType(), seen))
      return true;
  }
  return false;
}

/// Determine if the specified type requires laundering by checking if it is a
/// dynamic class type or contains a subobject which is a dynamic class type.
static bool typeRequiresBuiltinLaunder(clang::CIRGen::CIRGenModule &cgm,
                                       QualType ty) {
  if (!cgm.getCodeGenOpts().StrictVTablePointers)
    return false;
  llvm::SmallPtrSet<const Decl *, 16> seen;
  return typeRequiresBuiltinLaunderImp(cgm.getASTContext(), ty, seen);
}

RValue CIRGenFunction::emitRotate(const CallExpr *E, bool IsRotateRight) {
  auto src = emitScalarExpr(E->getArg(0));
  auto shiftAmt = emitScalarExpr(E->getArg(1));

  // The builtin's shift arg may have a different type than the source arg and
  // result, but the CIR ops uses the same type for all values.
  auto ty = src.getType();
  shiftAmt = builder.createIntCast(shiftAmt, ty);
  auto r =
      builder.create<cir::RotateOp>(getLoc(E->getSourceRange()), src, shiftAmt);
  if (!IsRotateRight)
    r->setAttr("left", mlir::UnitAttr::get(src.getContext()));
  return RValue::get(r);
}

static bool isMemBuiltinOutOfBoundPossible(const clang::Expr *sizeArg,
                                           const clang::Expr *dstSizeArg,
                                           clang::ASTContext &astContext,
                                           llvm::APSInt &size) {
  clang::Expr::EvalResult sizeResult, dstSizeResult;
  if (!sizeArg->EvaluateAsInt(sizeResult, astContext) ||
      !dstSizeArg->EvaluateAsInt(dstSizeResult, astContext))
    return true;
  size = sizeResult.Val.getInt();
  llvm::APSInt dstSize = dstSizeResult.Val.getInt();
  return size.ugt(dstSize);
}

RValue CIRGenFunction::emitBuiltinExpr(const GlobalDecl GD, unsigned BuiltinID,
                                       const CallExpr *E,
                                       ReturnValueSlot ReturnValue) {
  const FunctionDecl *FD = GD.getDecl()->getAsFunction();

  // See if we can constant fold this builtin.  If so, don't emit it at all.
  // TODO: Extend this handling to all builtin calls that we can constant-fold.
  Expr::EvalResult Result;
  if (E->isPRValue() && E->EvaluateAsRValue(Result, CGM.getASTContext()) &&
      !Result.hasSideEffects()) {
    if (Result.Val.isInt()) {
      return RValue::get(builder.getConstInt(getLoc(E->getSourceRange()),
                                             Result.Val.getInt()));
    }
    if (Result.Val.isFloat()) {
      // Note: we are using result type of CallExpr to determine the type of
      // the constant. Clang Codegen uses the result value to make judgement
      // of the type. We feel it should be Ok to use expression type because
      // it is hard to imagine a builtin function evaluates to
      // a value that over/underflows its own defined type.
      mlir::Type resTy = convertType(E->getType());
      return RValue::get(builder.getConstFP(getLoc(E->getExprLoc()), resTy,
                                            Result.Val.getFloat()));
    }
  }

  // If current long-double semantics is IEEE 128-bit, replace math builtins
  // of long-double with f128 equivalent.
  // TODO: This mutation should also be applied to other targets other than PPC,
  // after backend supports IEEE 128-bit style libcalls.
  if (getTarget().getTriple().isPPC64() &&
      &getTarget().getLongDoubleFormat() == &llvm::APFloat::IEEEquad())
    llvm_unreachable("NYI");

  // If the builtin has been declared explicitly with an assembler label,
  // disable the specialized emitting below. Ideally we should communicate the
  // rename in IR, or at least avoid generating the intrinsic calls that are
  // likely to get lowered to the renamed library functions.
  const unsigned BuiltinIDIfNoAsmLabel =
      FD->hasAttr<AsmLabelAttr>() ? 0 : BuiltinID;

  std::optional<bool> ErrnoOverriden;
  // ErrnoOverriden is true if math-errno is overriden via the
  // '#pragma float_control(precise, on)'. This pragma disables fast-math,
  // which implies math-errno.
  if (E->hasStoredFPFeatures()) {
    FPOptionsOverride OP = E->getFPFeatures();
    if (OP.hasMathErrnoOverride())
      ErrnoOverriden = OP.getMathErrnoOverride();
  }
  // True if 'atttibute__((optnone)) is used. This attibute overrides
  // fast-math which implies math-errno.
  bool OptNone = CurFuncDecl && CurFuncDecl->hasAttr<OptimizeNoneAttr>();

  // True if we are compiling at -O2 and errno has been disabled
  // using the '#pragma float_control(precise, off)', and
  // attribute opt-none hasn't been seen.
  [[maybe_unused]] bool ErrnoOverridenToFalseWithOpt =
      ErrnoOverriden.has_value() && !ErrnoOverriden.value() && !OptNone &&
      CGM.getCodeGenOpts().OptimizationLevel != 0;

  // There are LLVM math intrinsics/instructions corresponding to math library
  // functions except the LLVM op will never set errno while the math library
  // might. Also, math builtins have the same semantics as their math library
  // twins. Thus, we can transform math library and builtin calls to their
  // LLVM counterparts if the call is marked 'const' (known to never set errno).
  // In case FP exceptions are enabled, the experimental versions of the
  // intrinsics model those.
  [[maybe_unused]] bool ConstAlways =
      getContext().BuiltinInfo.isConst(BuiltinID);

  // There's a special case with the fma builtins where they are always const
  // if the target environment is GNU or the target is OS is Windows and we're
  // targeting the MSVCRT.dll environment.
  // FIXME: This list can be become outdated. Need to find a way to get it some
  // other way.
  switch (BuiltinID) {
  case Builtin::BI__builtin_fma:
  case Builtin::BI__builtin_fmaf:
  case Builtin::BI__builtin_fmal:
  case Builtin::BIfma:
  case Builtin::BIfmaf:
  case Builtin::BIfmal: {
    auto &Trip = CGM.getTriple();
    if (Trip.isGNUEnvironment() || Trip.isOSMSVCRT())
      ConstAlways = true;
    break;
  }
  case Builtin::BI__builtin_fmaf16:
    llvm_unreachable("Builtin::BI__builtin_fmaf16 NYI");
    break;
  default:
    break;
  }

  bool ConstWithoutErrnoAndExceptions =
      getContext().BuiltinInfo.isConstWithoutErrnoAndExceptions(BuiltinID);
  bool ConstWithoutExceptions =
      getContext().BuiltinInfo.isConstWithoutExceptions(BuiltinID);

  // ConstAttr is enabled in fast-math mode. In fast-math mode, math-errno is
  // disabled.
  // Math intrinsics are generated only when math-errno is disabled. Any pragmas
  // or attributes that affect math-errno should prevent or allow math
  // intrincs to be generated. Intrinsics are generated:
  //   1- In fast math mode, unless math-errno is overriden
  //      via '#pragma float_control(precise, on)', or via an
  //      'attribute__((optnone))'.
  //   2- If math-errno was enabled on command line but overriden
  //      to false via '#pragma float_control(precise, off))' and
  //      'attribute__((optnone))' hasn't been used.
  //   3- If we are compiling with optimization and errno has been disabled
  //      via '#pragma float_control(precise, off)', and
  //      'attribute__((optnone))' hasn't been used.

  bool ConstWithoutErrnoOrExceptions =
      ConstWithoutErrnoAndExceptions || ConstWithoutExceptions;
  bool GenerateIntrinsics =
      (ConstAlways && !OptNone) ||
      (!getLangOpts().MathErrno &&
       !(ErrnoOverriden.has_value() && ErrnoOverriden.value()) && !OptNone);
  if (!GenerateIntrinsics) {
    GenerateIntrinsics =
        ConstWithoutErrnoOrExceptions && !ConstWithoutErrnoAndExceptions;
    if (!GenerateIntrinsics)
      GenerateIntrinsics =
          ConstWithoutErrnoOrExceptions &&
          (!getLangOpts().MathErrno &&
           !(ErrnoOverriden.has_value() && ErrnoOverriden.value()) && !OptNone);
    if (!GenerateIntrinsics)
      GenerateIntrinsics =
          ConstWithoutErrnoOrExceptions && ErrnoOverridenToFalseWithOpt;
  }

  if (GenerateIntrinsics) {
    switch (BuiltinIDIfNoAsmLabel) {
    case Builtin::BIacos:
    case Builtin::BIacosf:
    case Builtin::BIacosl:
    case Builtin::BI__builtin_acos:
    case Builtin::BI__builtin_acosf:
    case Builtin::BI__builtin_acosf16:
    case Builtin::BI__builtin_acosl:
    case Builtin::BI__builtin_acosf128:
      llvm_unreachable("Builtin::BIacos like NYI");

    case Builtin::BIasin:
    case Builtin::BIasinf:
    case Builtin::BIasinl:
    case Builtin::BI__builtin_asin:
    case Builtin::BI__builtin_asinf:
    case Builtin::BI__builtin_asinf16:
    case Builtin::BI__builtin_asinl:
    case Builtin::BI__builtin_asinf128:
      llvm_unreachable("Builtin::BIasin like NYI");

    case Builtin::BIatan:
    case Builtin::BIatanf:
    case Builtin::BIatanl:
    case Builtin::BI__builtin_atan:
    case Builtin::BI__builtin_atanf:
    case Builtin::BI__builtin_atanf16:
    case Builtin::BI__builtin_atanl:
    case Builtin::BI__builtin_atanf128:
      llvm_unreachable("Builtin::BIatan like NYI");

    case Builtin::BIceil:
    case Builtin::BIceilf:
    case Builtin::BIceill:
    case Builtin::BI__builtin_ceil:
    case Builtin::BI__builtin_ceilf:
    case Builtin::BI__builtin_ceilf16:
    case Builtin::BI__builtin_ceill:
    case Builtin::BI__builtin_ceilf128:
      return emitUnaryMaybeConstrainedFPBuiltin<cir::CeilOp>(*this, *E);

    case Builtin::BIcopysign:
    case Builtin::BIcopysignf:
    case Builtin::BIcopysignl:
    case Builtin::BI__builtin_copysign:
    case Builtin::BI__builtin_copysignf:
    case Builtin::BI__builtin_copysignl:
      return emitBinaryFPBuiltin<cir::CopysignOp>(*this, *E);

    case Builtin::BI__builtin_copysignf16:
    case Builtin::BI__builtin_copysignf128:
      llvm_unreachable("BI__builtin_copysignf16 like NYI");

    case Builtin::BIcos:
    case Builtin::BIcosf:
    case Builtin::BIcosl:
    case Builtin::BI__builtin_cos:
    case Builtin::BI__builtin_cosf:
    case Builtin::BI__builtin_cosf16:
    case Builtin::BI__builtin_cosl:
    case Builtin::BI__builtin_cosf128:
      assert(!cir::MissingFeatures::fastMathFlags());
      return emitUnaryMaybeConstrainedFPBuiltin<cir::CosOp>(*this, *E);

    case Builtin::BIcosh:
    case Builtin::BIcoshf:
    case Builtin::BIcoshl:
    case Builtin::BI__builtin_cosh:
    case Builtin::BI__builtin_coshf:
    case Builtin::BI__builtin_coshf16:
    case Builtin::BI__builtin_coshl:
    case Builtin::BI__builtin_coshf128:
      llvm_unreachable("Builtin::BIcosh like NYI");

    case Builtin::BIexp:
    case Builtin::BIexpf:
    case Builtin::BIexpl:
    case Builtin::BI__builtin_exp:
    case Builtin::BI__builtin_expf:
    case Builtin::BI__builtin_expf16:
    case Builtin::BI__builtin_expl:
    case Builtin::BI__builtin_expf128:
      assert(!cir::MissingFeatures::fastMathFlags());
      return emitUnaryMaybeConstrainedFPBuiltin<cir::ExpOp>(*this, *E);

    case Builtin::BIexp2:
    case Builtin::BIexp2f:
    case Builtin::BIexp2l:
    case Builtin::BI__builtin_exp2:
    case Builtin::BI__builtin_exp2f:
    case Builtin::BI__builtin_exp2f16:
    case Builtin::BI__builtin_exp2l:
    case Builtin::BI__builtin_exp2f128:
      assert(!cir::MissingFeatures::fastMathFlags());
      return emitUnaryMaybeConstrainedFPBuiltin<cir::Exp2Op>(*this, *E);

    case Builtin::BI__builtin_exp10:
    case Builtin::BI__builtin_exp10f:
    case Builtin::BI__builtin_exp10f16:
    case Builtin::BI__builtin_exp10l:
    case Builtin::BI__builtin_exp10f128:
      llvm_unreachable("BI__builtin_exp10 like NYI");

    case Builtin::BIfabs:
    case Builtin::BIfabsf:
    case Builtin::BIfabsl:
    case Builtin::BI__builtin_fabs:
    case Builtin::BI__builtin_fabsf:
    case Builtin::BI__builtin_fabsf16:
    case Builtin::BI__builtin_fabsl:
    case Builtin::BI__builtin_fabsf128:
      return emitUnaryMaybeConstrainedFPBuiltin<cir::FAbsOp>(*this, *E);

    case Builtin::BIfloor:
    case Builtin::BIfloorf:
    case Builtin::BIfloorl:
    case Builtin::BI__builtin_floor:
    case Builtin::BI__builtin_floorf:
    case Builtin::BI__builtin_floorf16:
    case Builtin::BI__builtin_floorl:
    case Builtin::BI__builtin_floorf128:
      return emitUnaryMaybeConstrainedFPBuiltin<cir::FloorOp>(*this, *E);

    case Builtin::BIfma:
    case Builtin::BIfmaf:
    case Builtin::BIfmal:
    case Builtin::BI__builtin_fma:
    case Builtin::BI__builtin_fmaf:
    case Builtin::BI__builtin_fmaf16:
    case Builtin::BI__builtin_fmal:
    case Builtin::BI__builtin_fmaf128:
      llvm_unreachable("Builtin::BIfma like NYI");

    case Builtin::BIfmax:
    case Builtin::BIfmaxf:
    case Builtin::BIfmaxl:
    case Builtin::BI__builtin_fmax:
    case Builtin::BI__builtin_fmaxf:
    case Builtin::BI__builtin_fmaxl:
      return RValue::get(
          emitBinaryMaybeConstrainedFPBuiltin<cir::FMaxNumOp>(*this, *E));

    case Builtin::BI__builtin_fmaxf16:
    case Builtin::BI__builtin_fmaxf128:
      llvm_unreachable("BI__builtin_fmaxf16 like NYI");

    case Builtin::BIfmin:
    case Builtin::BIfminf:
    case Builtin::BIfminl:
    case Builtin::BI__builtin_fmin:
    case Builtin::BI__builtin_fminf:
    case Builtin::BI__builtin_fminl:
      return RValue::get(
          emitBinaryMaybeConstrainedFPBuiltin<cir::FMinNumOp>(*this, *E));

    case Builtin::BI__builtin_fminf16:
    case Builtin::BI__builtin_fminf128:
      llvm_unreachable("BI__builtin_fminf16 like NYI");

    // fmod() is a special-case. It maps to the frem instruction rather than an
    // LLVM intrinsic.
    case Builtin::BIfmod:
    case Builtin::BIfmodf:
    case Builtin::BIfmodl:
    case Builtin::BI__builtin_fmod:
    case Builtin::BI__builtin_fmodf:
    case Builtin::BI__builtin_fmodl:
      assert(!cir::MissingFeatures::fastMathFlags());
      return emitBinaryFPBuiltin<cir::FModOp>(*this, *E);

    case Builtin::BI__builtin_fmodf16:
    case Builtin::BI__builtin_fmodf128:
    case Builtin::BI__builtin_elementwise_fmod:
      llvm_unreachable("BI__builtin_fmodf16 like NYI");

    case Builtin::BIlog:
    case Builtin::BIlogf:
    case Builtin::BIlogl:
    case Builtin::BI__builtin_log:
    case Builtin::BI__builtin_logf:
    case Builtin::BI__builtin_logf16:
    case Builtin::BI__builtin_logl:
    case Builtin::BI__builtin_logf128:
      assert(!cir::MissingFeatures::fastMathFlags());
      return emitUnaryMaybeConstrainedFPBuiltin<cir::LogOp>(*this, *E);

    case Builtin::BIlog10:
    case Builtin::BIlog10f:
    case Builtin::BIlog10l:
    case Builtin::BI__builtin_log10:
    case Builtin::BI__builtin_log10f:
    case Builtin::BI__builtin_log10f16:
    case Builtin::BI__builtin_log10l:
    case Builtin::BI__builtin_log10f128:
      assert(!cir::MissingFeatures::fastMathFlags());
      return emitUnaryMaybeConstrainedFPBuiltin<cir::Log10Op>(*this, *E);

    case Builtin::BIlog2:
    case Builtin::BIlog2f:
    case Builtin::BIlog2l:
    case Builtin::BI__builtin_log2:
    case Builtin::BI__builtin_log2f:
    case Builtin::BI__builtin_log2f16:
    case Builtin::BI__builtin_log2l:
    case Builtin::BI__builtin_log2f128:
      assert(!cir::MissingFeatures::fastMathFlags());
      return emitUnaryMaybeConstrainedFPBuiltin<cir::Log2Op>(*this, *E);

    case Builtin::BInearbyint:
    case Builtin::BInearbyintf:
    case Builtin::BInearbyintl:
    case Builtin::BI__builtin_nearbyint:
    case Builtin::BI__builtin_nearbyintf:
    case Builtin::BI__builtin_nearbyintl:
    case Builtin::BI__builtin_nearbyintf128:
      return emitUnaryMaybeConstrainedFPBuiltin<cir::NearbyintOp>(*this, *E);

    case Builtin::BIpow:
    case Builtin::BIpowf:
    case Builtin::BIpowl:
    case Builtin::BI__builtin_pow:
    case Builtin::BI__builtin_powf:
    case Builtin::BI__builtin_powl:
      assert(!cir::MissingFeatures::fastMathFlags());
      return RValue::get(
          emitBinaryMaybeConstrainedFPBuiltin<cir::PowOp>(*this, *E));

    case Builtin::BI__builtin_powf16:
    case Builtin::BI__builtin_powf128:
      llvm_unreachable("BI__builtin_powf16 like NYI");

    case Builtin::BIrint:
    case Builtin::BIrintf:
    case Builtin::BIrintl:
    case Builtin::BI__builtin_rint:
    case Builtin::BI__builtin_rintf:
    case Builtin::BI__builtin_rintf16:
    case Builtin::BI__builtin_rintl:
    case Builtin::BI__builtin_rintf128:
      return emitUnaryMaybeConstrainedFPBuiltin<cir::RintOp>(*this, *E);

    case Builtin::BIround:
    case Builtin::BIroundf:
    case Builtin::BIroundl:
    case Builtin::BI__builtin_round:
    case Builtin::BI__builtin_roundf:
    case Builtin::BI__builtin_roundf16:
    case Builtin::BI__builtin_roundl:
    case Builtin::BI__builtin_roundf128:
      return emitUnaryMaybeConstrainedFPBuiltin<cir::RoundOp>(*this, *E);

    case Builtin::BIroundeven:
    case Builtin::BIroundevenf:
    case Builtin::BIroundevenl:
    case Builtin::BI__builtin_roundeven:
    case Builtin::BI__builtin_roundevenf:
    case Builtin::BI__builtin_roundevenf16:
    case Builtin::BI__builtin_roundevenl:
    case Builtin::BI__builtin_roundevenf128:
      llvm_unreachable("Builtin::BIroundeven like NYI");

    case Builtin::BIsin:
    case Builtin::BIsinf:
    case Builtin::BIsinl:
    case Builtin::BI__builtin_sin:
    case Builtin::BI__builtin_sinf:
    case Builtin::BI__builtin_sinf16:
    case Builtin::BI__builtin_sinl:
    case Builtin::BI__builtin_sinf128:
      assert(!cir::MissingFeatures::fastMathFlags());
      return emitUnaryMaybeConstrainedFPBuiltin<cir::SinOp>(*this, *E);

    case Builtin::BIsqrt:
    case Builtin::BIsqrtf:
    case Builtin::BIsqrtl:
    case Builtin::BI__builtin_sqrt:
    case Builtin::BI__builtin_sqrtf:
    case Builtin::BI__builtin_sqrtf16:
    case Builtin::BI__builtin_sqrtl:
    case Builtin::BI__builtin_sqrtf128:
      assert(!cir::MissingFeatures::fastMathFlags());
      return emitUnaryMaybeConstrainedFPBuiltin<cir::SqrtOp>(*this, *E);

    case Builtin::BI__builtin_elementwise_sqrt:
      llvm_unreachable("BI__builtin_elementwise_sqrt NYI");

    case Builtin::BItan:
    case Builtin::BItanf:
    case Builtin::BItanl:
    case Builtin::BI__builtin_tan:
    case Builtin::BI__builtin_tanf:
    case Builtin::BI__builtin_tanf16:
    case Builtin::BI__builtin_tanl:
    case Builtin::BI__builtin_tanf128:
      llvm_unreachable("Builtin::BItan like NYI");

    case Builtin::BItanh:
    case Builtin::BItanhf:
    case Builtin::BItanhl:
    case Builtin::BI__builtin_tanh:
    case Builtin::BI__builtin_tanhf:
    case Builtin::BI__builtin_tanhf16:
    case Builtin::BI__builtin_tanhl:
    case Builtin::BI__builtin_tanhf128:
      llvm_unreachable("Builtin::BItanh like NYI");

    case Builtin::BItrunc:
    case Builtin::BItruncf:
    case Builtin::BItruncl:
    case Builtin::BI__builtin_trunc:
    case Builtin::BI__builtin_truncf:
    case Builtin::BI__builtin_truncf16:
    case Builtin::BI__builtin_truncl:
    case Builtin::BI__builtin_truncf128:
      return emitUnaryMaybeConstrainedFPBuiltin<cir::TruncOp>(*this, *E);

    case Builtin::BIlround:
    case Builtin::BIlroundf:
    case Builtin::BIlroundl:
    case Builtin::BI__builtin_lround:
    case Builtin::BI__builtin_lroundf:
    case Builtin::BI__builtin_lroundl:
      return emitUnaryMaybeConstrainedFPToIntBuiltin<cir::LroundOp>(*this, *E);

    case Builtin::BI__builtin_lroundf128:
      llvm_unreachable("BI__builtin_lroundf128 NYI");

    case Builtin::BIllround:
    case Builtin::BIllroundf:
    case Builtin::BIllroundl:
    case Builtin::BI__builtin_llround:
    case Builtin::BI__builtin_llroundf:
    case Builtin::BI__builtin_llroundl:
      return emitUnaryMaybeConstrainedFPToIntBuiltin<cir::LLroundOp>(*this, *E);

    case Builtin::BI__builtin_llroundf128:
      llvm_unreachable("BI__builtin_llroundf128 NYI");

    case Builtin::BIlrint:
    case Builtin::BIlrintf:
    case Builtin::BIlrintl:
    case Builtin::BI__builtin_lrint:
    case Builtin::BI__builtin_lrintf:
    case Builtin::BI__builtin_lrintl:
      return emitUnaryMaybeConstrainedFPToIntBuiltin<cir::LrintOp>(*this, *E);

    case Builtin::BI__builtin_lrintf128:
      llvm_unreachable("BI__builtin_lrintf128 NYI");

    case Builtin::BIllrint:
    case Builtin::BIllrintf:
    case Builtin::BIllrintl:
    case Builtin::BI__builtin_llrint:
    case Builtin::BI__builtin_llrintf:
    case Builtin::BI__builtin_llrintl:
      return emitUnaryMaybeConstrainedFPToIntBuiltin<cir::LLrintOp>(*this, *E);

    case Builtin::BI__builtin_llrintf128:
      llvm_unreachable("BI__builtin_llrintf128 NYI");

    case Builtin::BI__builtin_ldexp:
    case Builtin::BI__builtin_ldexpf:
    case Builtin::BI__builtin_ldexpl:
    case Builtin::BI__builtin_ldexpf16:
    case Builtin::BI__builtin_ldexpf128:
      llvm_unreachable("Builtin::BI__builtin_ldexp NYI");

    default:
      break;
    }
  }

  switch (BuiltinIDIfNoAsmLabel) {
  default:
    break;

  case Builtin::BI__builtin___CFStringMakeConstantString:
  case Builtin::BI__builtin___NSStringMakeConstantString:
    llvm_unreachable("BI__builtin___CFStringMakeConstantString like NYI");

    // C stdarg builtins.
  case Builtin::BI__builtin_stdarg_start:
  case Builtin::BI__builtin_va_start:
  case Builtin::BI__va_start:
  case Builtin::BI__builtin_va_end: {
    emitVAStartEnd(BuiltinID == Builtin::BI__va_start
                       ? emitScalarExpr(E->getArg(0))
                       : emitVAListRef(E->getArg(0)).getPointer(),
                   BuiltinID != Builtin::BI__builtin_va_end);
    return {};
  }
  case Builtin::BI__builtin_va_copy: {
    auto dstPtr = emitVAListRef(E->getArg(0)).getPointer();
    auto srcPtr = emitVAListRef(E->getArg(1)).getPointer();
    builder.create<cir::VACopyOp>(dstPtr.getLoc(), dstPtr, srcPtr);
    return {};
  }

  case Builtin::BIabs:
  case Builtin::BIlabs:
  case Builtin::BIllabs:
  case Builtin::BI__builtin_abs:
  case Builtin::BI__builtin_labs:
  case Builtin::BI__builtin_llabs: {
    bool SanitizeOverflow = SanOpts.has(SanitizerKind::SignedIntegerOverflow);
    auto Arg = emitScalarExpr(E->getArg(0));
    mlir::Value Result;
    switch (getLangOpts().getSignedOverflowBehavior()) {
    case LangOptions::SOB_Defined: {
      auto Call = getBuilder().create<cir::AbsOp>(getLoc(E->getExprLoc()),
                                                  Arg.getType(), Arg, false);
      Result = Call->getResult(0);
      break;
    }
    case LangOptions::SOB_Undefined: {
      if (!SanitizeOverflow) {
        auto Call = getBuilder().create<cir::AbsOp>(getLoc(E->getExprLoc()),
                                                    Arg.getType(), Arg, true);
        Result = Call->getResult(0);
        break;
      }
      llvm_unreachable("BI__builtin_abs with LangOptions::SOB_Undefined when "
                       "SanitizeOverflow is true");
    }
      [[fallthrough]];
    case LangOptions::SOB_Trapping:
      llvm_unreachable("BI__builtin_abs with LangOptions::SOB_Trapping");
    }
    return RValue::get(Result);
  }
  case Builtin::BI__builtin_complex: {
    mlir::Value Real = emitScalarExpr(E->getArg(0));
    mlir::Value Imag = emitScalarExpr(E->getArg(1));
    mlir::Value Complex =
        builder.createComplexCreate(getLoc(E->getExprLoc()), Real, Imag);
    return RValue::getComplex(Complex);
  }

  case Builtin::BI__builtin_conj:
  case Builtin::BI__builtin_conjf:
  case Builtin::BI__builtin_conjl:
  case Builtin::BIconj:
  case Builtin::BIconjf:
  case Builtin::BIconjl: {
    mlir::Value ComplexVal = emitComplexExpr(E->getArg(0));
    mlir::Value Conj = builder.createUnaryOp(getLoc(E->getExprLoc()),
                                             cir::UnaryOpKind::Not, ComplexVal);
    return RValue::getComplex(Conj);
  }

  case Builtin::BI__builtin_creal:
  case Builtin::BI__builtin_crealf:
  case Builtin::BI__builtin_creall:
  case Builtin::BIcreal:
  case Builtin::BIcrealf:
  case Builtin::BIcreall: {
    mlir::Value ComplexVal = emitComplexExpr(E->getArg(0));
    mlir::Value Real =
        builder.createComplexReal(getLoc(E->getExprLoc()), ComplexVal);
    return RValue::get(Real);
  }

  case Builtin::BI__builtin_preserve_access_index:
    llvm_unreachable("Builtin::BI__builtin_preserve_access_index NYI");

  case Builtin::BI__builtin_cimag:
  case Builtin::BI__builtin_cimagf:
  case Builtin::BI__builtin_cimagl:
  case Builtin::BIcimag:
  case Builtin::BIcimagf:
  case Builtin::BIcimagl: {
    mlir::Value ComplexVal = emitComplexExpr(E->getArg(0));
    mlir::Value Real =
        builder.createComplexImag(getLoc(E->getExprLoc()), ComplexVal);
    return RValue::get(Real);
  }

  case Builtin::BI__builtin_clrsb:
  case Builtin::BI__builtin_clrsbl:
  case Builtin::BI__builtin_clrsbll:
    return emitBuiltinBitOp<cir::BitClrsbOp>(*this, E, std::nullopt);

  case Builtin::BI__builtin_ctzs:
  case Builtin::BI__builtin_ctz:
  case Builtin::BI__builtin_ctzl:
  case Builtin::BI__builtin_ctzll:
  case Builtin::BI__builtin_ctzg:
    return emitBuiltinBitOp<cir::BitCtzOp>(*this, E, BCK_CTZPassedZero);

  case Builtin::BI__builtin_clzs:
  case Builtin::BI__builtin_clz:
  case Builtin::BI__builtin_clzl:
  case Builtin::BI__builtin_clzll:
  case Builtin::BI__builtin_clzg:
    return emitBuiltinBitOp<cir::BitClzOp>(*this, E, BCK_CLZPassedZero);

  case Builtin::BI__builtin_ffs:
  case Builtin::BI__builtin_ffsl:
  case Builtin::BI__builtin_ffsll:
    return emitBuiltinBitOp<cir::BitFfsOp>(*this, E, std::nullopt);

  case Builtin::BI__builtin_parity:
  case Builtin::BI__builtin_parityl:
  case Builtin::BI__builtin_parityll:
    return emitBuiltinBitOp<cir::BitParityOp>(*this, E, std::nullopt);

  case Builtin::BI__lzcnt16:
  case Builtin::BI__lzcnt:
  case Builtin::BI__lzcnt64:
    llvm_unreachable("BI__lzcnt16 like NYI");

  case Builtin::BI__popcnt16:
  case Builtin::BI__popcnt:
  case Builtin::BI__popcnt64:
  case Builtin::BI__builtin_popcount:
  case Builtin::BI__builtin_popcountl:
  case Builtin::BI__builtin_popcountll:
  case Builtin::BI__builtin_popcountg:
    return emitBuiltinBitOp<cir::BitPopcountOp>(*this, E, std::nullopt);

  case Builtin::BI__builtin_unpredictable: {
    if (CGM.getCodeGenOpts().OptimizationLevel != 0)
      assert(!cir::MissingFeatures::insertBuiltinUnpredictable());
    return RValue::get(emitScalarExpr(E->getArg(0)));
  }

  case Builtin::BI__builtin_expect:
  case Builtin::BI__builtin_expect_with_probability: {
    auto ArgValue = emitScalarExpr(E->getArg(0));
    auto ExpectedValue = emitScalarExpr(E->getArg(1));

    // Don't generate cir.expect on -O0 as the backend won't use it for
    // anything. Note, we still IRGen ExpectedValue because it could have
    // side-effects.
    if (CGM.getCodeGenOpts().OptimizationLevel == 0)
      return RValue::get(ArgValue);

    mlir::FloatAttr ProbAttr = {};
    if (BuiltinIDIfNoAsmLabel == Builtin::BI__builtin_expect_with_probability) {
      llvm::APFloat Probability(0.0);
      const Expr *ProbArg = E->getArg(2);
      bool EvalSucceed =
          ProbArg->EvaluateAsFloat(Probability, CGM.getASTContext());
      assert(EvalSucceed && "probability should be able to evaluate as float");
      (void)EvalSucceed;
      bool LoseInfo = false;
      Probability.convert(llvm::APFloat::IEEEdouble(),
                          llvm::RoundingMode::Dynamic, &LoseInfo);
      ProbAttr = mlir::FloatAttr::get(mlir::Float64Type::get(&getMLIRContext()),
                                      Probability);
    }

    auto result = builder.create<cir::ExpectOp>(getLoc(E->getSourceRange()),
                                                ArgValue.getType(), ArgValue,
                                                ExpectedValue, ProbAttr);

    return RValue::get(result);
  }

  case Builtin::BI__builtin_assume_aligned: {
    const Expr *ptr = E->getArg(0);
    mlir::Value ptrValue = emitScalarExpr(ptr);
    mlir::Value offsetValue =
        (E->getNumArgs() > 2) ? emitScalarExpr(E->getArg(2)) : nullptr;

    mlir::Attribute alignmentAttr = ConstantEmitter(*this).emitAbstract(
        E->getArg(1), E->getArg(1)->getType());
    std::int64_t alignment = cast<cir::IntAttr>(alignmentAttr).getSInt();

    ptrValue = emitAlignmentAssumption(ptrValue, ptr, ptr->getExprLoc(),
                                       builder.getI64IntegerAttr(alignment),
                                       offsetValue);
    return RValue::get(ptrValue);
  }

  case Builtin::BI__assume:
  case Builtin::BI__builtin_assume: {
    if (E->getArg(0)->HasSideEffects(getContext()))
      return RValue::get(nullptr);

    mlir::Value argValue = emitCheckedArgForAssume(E->getArg(0));
    builder.create<cir::AssumeOp>(getLoc(E->getExprLoc()), argValue);
    return RValue::get(nullptr);
  }

  case Builtin::BI__builtin_assume_separate_storage: {
    const Expr *arg0 = E->getArg(0);
    const Expr *arg1 = E->getArg(1);

    mlir::Value value0 = emitScalarExpr(arg0);
    mlir::Value value1 = emitScalarExpr(arg1);

    builder.create<cir::AssumeSepStorageOp>(getLoc(E->getExprLoc()), value0,
                                            value1);
    return RValue::get(nullptr);
  }

  case Builtin::BI__builtin_allow_runtime_check:
    llvm_unreachable("BI__builtin_allow_runtime_check NYI");

  case Builtin::BI__arithmetic_fence:
    llvm_unreachable("BI__arithmetic_fence NYI");

  case Builtin::BI__builtin_bswap16:
  case Builtin::BI__builtin_bswap32:
  case Builtin::BI__builtin_bswap64:
  case Builtin::BI_byteswap_ushort:
  case Builtin::BI_byteswap_ulong:
  case Builtin::BI_byteswap_uint64: {
    auto arg = emitScalarExpr(E->getArg(0));
    return RValue::get(
        builder.create<cir::ByteswapOp>(getLoc(E->getSourceRange()), arg));
  }

  case Builtin::BI__builtin_bitreverse8:
  case Builtin::BI__builtin_bitreverse16:
  case Builtin::BI__builtin_bitreverse32:
  case Builtin::BI__builtin_bitreverse64: {
    mlir::Value arg = emitScalarExpr(E->getArg(0));
    return RValue::get(
        builder.create<cir::BitReverseOp>(getLoc(E->getSourceRange()), arg));
  }

  case Builtin::BI__builtin_rotateleft8:
  case Builtin::BI__builtin_rotateleft16:
  case Builtin::BI__builtin_rotateleft32:
  case Builtin::BI__builtin_rotateleft64:
  case Builtin::BI_rotl8: // Microsoft variants of rotate left
  case Builtin::BI_rotl16:
  case Builtin::BI_rotl:
  case Builtin::BI_lrotl:
  case Builtin::BI_rotl64:
    return emitRotate(E, false);

  case Builtin::BI__builtin_rotateright8:
  case Builtin::BI__builtin_rotateright16:
  case Builtin::BI__builtin_rotateright32:
  case Builtin::BI__builtin_rotateright64:
  case Builtin::BI_rotr8: // Microsoft variants of rotate right
  case Builtin::BI_rotr16:
  case Builtin::BI_rotr:
  case Builtin::BI_lrotr:
  case Builtin::BI_rotr64:
    return emitRotate(E, true);

  case Builtin::BI__builtin_constant_p: {
    mlir::Type ResultType = convertType(E->getType());

    const Expr *Arg = E->getArg(0);
    QualType ArgType = Arg->getType();
    // FIXME: The allowance for Obj-C pointers and block pointers is historical
    // and likely a mistake.
    if (!ArgType->isIntegralOrEnumerationType() && !ArgType->isFloatingType() &&
        !ArgType->isObjCObjectPointerType() && !ArgType->isBlockPointerType())
      // Per the GCC documentation, only numeric constants are recognized after
      // inlining.
      return RValue::get(
          builder.getConstInt(getLoc(E->getSourceRange()),
                              mlir::cast<cir::IntType>(ResultType), 0));

    if (Arg->HasSideEffects(getContext()))
      // The argument is unevaluated, so be conservative if it might have
      // side-effects.
      return RValue::get(
          builder.getConstInt(getLoc(E->getSourceRange()),
                              mlir::cast<cir::IntType>(ResultType), 0));

    mlir::Value ArgValue = emitScalarExpr(Arg);
    if (ArgType->isObjCObjectPointerType())
      // Convert Objective-C objects to id because we cannot distinguish between
      // LLVM types for Obj-C classes as they are opaque.
      ArgType = CGM.getASTContext().getObjCIdType();
    ArgValue = builder.createBitcast(ArgValue, convertType(ArgType));

    mlir::Value Result = builder.create<cir::IsConstantOp>(
        getLoc(E->getSourceRange()), ArgValue);
    if (Result.getType() != ResultType)
      Result = builder.createBoolToInt(Result, ResultType);
    return RValue::get(Result);
  }

  case Builtin::BI__builtin_dynamic_object_size: {
    // Fallthrough below, assert until we have a testcase.
    llvm_unreachable("BI__builtin_dynamic_object_size NYI");
  }
  case Builtin::BI__builtin_object_size: {
    unsigned Type =
        E->getArg(1)->EvaluateKnownConstInt(getContext()).getZExtValue();
    auto ResType = mlir::dyn_cast<cir::IntType>(convertType(E->getType()));
    assert(ResType && "not sure what to do?");

    // We pass this builtin onto the optimizer so that it can figure out the
    // object size in more complex cases.
    bool IsDynamic = BuiltinID == Builtin::BI__builtin_dynamic_object_size;
    return RValue::get(emitBuiltinObjectSize(E->getArg(0), Type, ResType,
                                             /*EmittedE=*/nullptr, IsDynamic));
  }

  case Builtin::BI__builtin_prefetch: {
    auto evaluateOperandAsInt = [&](const Expr *Arg) {
      Expr::EvalResult Res;
      [[maybe_unused]] bool EvalSucceed =
          Arg->EvaluateAsInt(Res, CGM.getASTContext());
      assert(EvalSucceed && "expression should be able to evaluate as int");
      return Res.Val.getInt().getZExtValue();
    };

    bool IsWrite = false;
    if (E->getNumArgs() > 1)
      IsWrite = evaluateOperandAsInt(E->getArg(1));

    int Locality = 0;
    if (E->getNumArgs() > 2)
      Locality = evaluateOperandAsInt(E->getArg(2));

    mlir::Value Address = emitScalarExpr(E->getArg(0));
    builder.create<cir::PrefetchOp>(getLoc(E->getSourceRange()), Address,
                                    Locality, IsWrite);
    return RValue::get(nullptr);
  }
  case Builtin::BI__builtin_readcyclecounter:
    llvm_unreachable("BI__builtin_readcyclecounter NYI");
  case Builtin::BI__builtin_readsteadycounter:
    llvm_unreachable("BI__builtin_readsteadycounter NYI");

  case Builtin::BI__builtin___clear_cache: {
    mlir::Type voidTy = cir::VoidType::get(&getMLIRContext());
    mlir::Value begin =
        builder.createPtrBitcast(emitScalarExpr(E->getArg(0)), voidTy);
    mlir::Value end =
        builder.createPtrBitcast(emitScalarExpr(E->getArg(1)), voidTy);
    builder.create<cir::ClearCacheOp>(getLoc(E->getSourceRange()), begin, end);
    return RValue::get(nullptr);
  }
  case Builtin::BI__builtin_trap: {
    builder.create<cir::TrapOp>(getLoc(E->getExprLoc()));

    // Note that cir.trap is a terminator so we need to start a new block to
    // preserve the insertion point.
    builder.createBlock(builder.getBlock()->getParent());

    return RValue::get(nullptr);
  }
  case Builtin::BI__builtin_verbose_trap:
    llvm_unreachable("BI__builtin_verbose_trap NYI");
  case Builtin::BI__debugbreak:
    llvm_unreachable("BI__debugbreak NYI");
  case Builtin::BI__builtin_unreachable: {
    emitUnreachable(E->getExprLoc());

    // We do need to preserve an insertion point.
    builder.createBlock(builder.getBlock()->getParent());

    return RValue::get(nullptr);
  }

  case Builtin::BI__builtin_powi:
  case Builtin::BI__builtin_powif:
  case Builtin::BI__builtin_powil:
    llvm_unreachable("BI__builtin_powi like NYI");

  case Builtin::BI__builtin_frexp:
  case Builtin::BI__builtin_frexpf:
  case Builtin::BI__builtin_frexpf128:
  case Builtin::BI__builtin_frexpf16:
    llvm_unreachable("BI__builtin_frexp like NYI");

  case Builtin::BI__builtin_isgreater:
  case Builtin::BI__builtin_isgreaterequal:
  case Builtin::BI__builtin_isless:
  case Builtin::BI__builtin_islessequal:
  case Builtin::BI__builtin_islessgreater:
  case Builtin::BI__builtin_isunordered:
    llvm_unreachable("BI__builtin_isgreater and BI__builtin_isless like NYI");

  case Builtin::BI__builtin_nondeterministic_value:
    llvm_unreachable("BI__builtin_nondeterministic_value NYI");

  case Builtin::BI__builtin_elementwise_abs: {
    mlir::Type cirTy = convertType(E->getArg(0)->getType());
    bool isIntTy = cir::isIntOrIntVectorTy(cirTy);
    if (!isIntTy) {
      mlir::Type eltTy = cirTy;
      if (mlir::isa<cir::VectorType>(cirTy))
        eltTy = mlir::cast<cir::VectorType>(cirTy).getEltType();
      if (mlir::isa<cir::SingleType, cir::DoubleType>(eltTy)) {
        return emitUnaryMaybeConstrainedFPBuiltin<cir::FAbsOp>(*this, *E);
      }
      llvm_unreachable("unsupported type for BI__builtin_elementwise_abs");
    }
    mlir::Value arg = emitScalarExpr(E->getArg(0));
    auto call = getBuilder().create<cir::AbsOp>(getLoc(E->getExprLoc()),
                                                arg.getType(), arg, false);
    mlir::Value result = call->getResult(0);
    return RValue::get(result);
  }
  case Builtin::BI__builtin_elementwise_acos: {
    return emitBuiltinWithOneOverloadedType<1>(E, "acos");
  }
  case Builtin::BI__builtin_elementwise_asin:
    llvm_unreachable("BI__builtin_elementwise_asin NYI");
  case Builtin::BI__builtin_elementwise_atan:
    llvm_unreachable("BI__builtin_elementwise_atan NYI");
  case Builtin::BI__builtin_elementwise_atan2:
    llvm_unreachable("BI__builtin_elementwise_atan2 NYI");
  case Builtin::BI__builtin_elementwise_ceil:
    llvm_unreachable("BI__builtin_elementwise_ceil NYI");
  case Builtin::BI__builtin_elementwise_exp: {
    return emitUnaryFPBuiltin<cir::ExpOp>(*this, *E);
  }
  case Builtin::BI__builtin_elementwise_exp2:
    llvm_unreachable("BI__builtin_elementwise_exp2 NYI");
  case Builtin::BI__builtin_elementwise_log:
    llvm_unreachable("BI__builtin_elementwise_log NYI");
  case Builtin::BI__builtin_elementwise_log2:
    llvm_unreachable("BI__builtin_elementwise_log2 NYI");
  case Builtin::BI__builtin_elementwise_log10:
    llvm_unreachable("BI__builtin_elementwise_log10 NYI");
  case Builtin::BI__builtin_elementwise_pow:
    llvm_unreachable("BI__builtin_elementwise_pow NYI");
  case Builtin::BI__builtin_elementwise_bitreverse:
    llvm_unreachable("BI__builtin_elementwise_bitreverse NYI");
  case Builtin::BI__builtin_elementwise_cos:
    llvm_unreachable("BI__builtin_elementwise_cos NYI");
  case Builtin::BI__builtin_elementwise_cosh:
    llvm_unreachable("BI__builtin_elementwise_cosh NYI");
  case Builtin::BI__builtin_elementwise_floor:
    llvm_unreachable("BI__builtin_elementwise_floor NYI");
  case Builtin::BI__builtin_elementwise_popcount:
    llvm_unreachable("BI__builtin_elementwise_popcount NYI");
  case Builtin::BI__builtin_elementwise_roundeven:
    llvm_unreachable("BI__builtin_elementwise_roundeven NYI");
  case Builtin::BI__builtin_elementwise_round:
    llvm_unreachable("BI__builtin_elementwise_round NYI");
  case Builtin::BI__builtin_elementwise_rint:
    llvm_unreachable("BI__builtin_elementwise_rint NYI");
  case Builtin::BI__builtin_elementwise_nearbyint:
    llvm_unreachable("BI__builtin_elementwise_nearbyint NYI");
  case Builtin::BI__builtin_elementwise_sin:
    llvm_unreachable("BI__builtin_elementwise_sin NYI");
  case Builtin::BI__builtin_elementwise_sinh:
    llvm_unreachable("BI__builtin_elementwise_sinh NYI");
  case Builtin::BI__builtin_elementwise_tan:
    llvm_unreachable("BI__builtin_elementwise_tan NYI");
  case Builtin::BI__builtin_elementwise_tanh:
    llvm_unreachable("BI__builtin_elementwise_tanh NYI");
  case Builtin::BI__builtin_elementwise_trunc:
    llvm_unreachable("BI__builtin_elementwise_trunc NYI");
  case Builtin::BI__builtin_elementwise_canonicalize:
    llvm_unreachable("BI__builtin_elementwise_canonicalize NYI");
  case Builtin::BI__builtin_elementwise_copysign:
    llvm_unreachable("BI__builtin_elementwise_copysign NYI");
  case Builtin::BI__builtin_elementwise_fma:
    llvm_unreachable("BI__builtin_elementwise_fma NYI");
  case Builtin::BI__builtin_elementwise_add_sat:
  case Builtin::BI__builtin_elementwise_sub_sat:
    llvm_unreachable("BI__builtin_elementwise_add/sub_sat NYI");

  case Builtin::BI__builtin_elementwise_max:
    llvm_unreachable("BI__builtin_elementwise_max NYI");
  case Builtin::BI__builtin_elementwise_min:
    llvm_unreachable("BI__builtin_elementwise_min NYI");

  case Builtin::BI__builtin_elementwise_maximum:
    llvm_unreachable("BI__builtin_elementwise_maximum NYI");

  case Builtin::BI__builtin_elementwise_minimum:
    llvm_unreachable("BI__builtin_elementwise_minimum NYI");

  case Builtin::BI__builtin_reduce_max:
    llvm_unreachable("BI__builtin_reduce_max NYI");

  case Builtin::BI__builtin_reduce_min:
    llvm_unreachable("BI__builtin_reduce_min NYI");

  case Builtin::BI__builtin_reduce_add:
    llvm_unreachable("BI__builtin_reduce_add NYI");
  case Builtin::BI__builtin_reduce_mul:
    llvm_unreachable("BI__builtin_reduce_mul NYI");
  case Builtin::BI__builtin_reduce_xor:
    llvm_unreachable("BI__builtin_reduce_xor NYI");
  case Builtin::BI__builtin_reduce_or:
    llvm_unreachable("BI__builtin_reduce_or NYI");
  case Builtin::BI__builtin_reduce_and:
    llvm_unreachable("BI__builtin_reduce_and NYI");
  case Builtin::BI__builtin_reduce_maximum:
    llvm_unreachable("BI__builtin_reduce_maximum NYI");
  case Builtin::BI__builtin_reduce_minimum:
    llvm_unreachable("BI__builtin_reduce_minimum NYI");

  case Builtin::BI__builtin_matrix_transpose:
    llvm_unreachable("BI__builtin_matrix_transpose NYI");

  case Builtin::BI__builtin_matrix_column_major_load:
    llvm_unreachable("BI__builtin_matrix_column_major_load NYI");

  case Builtin::BI__builtin_matrix_column_major_store:
    llvm_unreachable("BI__builtin_matrix_column_major_store NYI");

  case Builtin::BI__builtin_isinf_sign: {
    CIRGenFunction::CIRGenFPOptionsRAII FPOptsRAII(*this, E);
    mlir::Location Loc = getLoc(E->getBeginLoc());
    mlir::Value Arg = emitScalarExpr(E->getArg(0));
    mlir::Value AbsArg = builder.create<cir::FAbsOp>(Loc, Arg.getType(), Arg);
    mlir::Value IsInf =
        builder.createIsFPClass(Loc, AbsArg, FPClassTest::fcInf);
    mlir::Value IsNeg = emitSignBit(Loc, *this, Arg);
    auto IntTy = convertType(E->getType());
    auto Zero = builder.getNullValue(IntTy, Loc);
    auto One = builder.getConstant(Loc, cir::IntAttr::get(IntTy, 1));
    auto NegativeOne = builder.getConstant(Loc, cir::IntAttr::get(IntTy, -1));
    auto SignResult = builder.createSelect(Loc, IsNeg, NegativeOne, One);
    auto Result = builder.createSelect(Loc, IsInf, SignResult, Zero);
    return RValue::get(Result);
  }

  case Builtin::BI__builtin_flt_rounds:
    llvm_unreachable("BI__builtin_flt_rounds NYI");

  case Builtin::BI__builtin_set_flt_rounds:
    llvm_unreachable("BI__builtin_set_flt_rounds NYI");

  case Builtin::BI__builtin_fpclassify:
    llvm_unreachable("BI__builtin_fpclassify NYI");

  case Builtin::BIalloca:
  case Builtin::BI_alloca:
  case Builtin::BI__builtin_alloca_uninitialized:
  case Builtin::BI__builtin_alloca: {
    // Get alloca size input
    mlir::Value Size = emitScalarExpr(E->getArg(0));

    // The alignment of the alloca should correspond to __BIGGEST_ALIGNMENT__.
    const TargetInfo &TI = getContext().getTargetInfo();
    const CharUnits SuitableAlignmentInBytes =
        getContext().toCharUnitsFromBits(TI.getSuitableAlign());

    // Emit the alloca op with type `u8 *` to match the semantics of
    // `llvm.alloca`. We later bitcast the type to `void *` to match the
    // semantics of C/C++
    // FIXME(cir): It may make sense to allow AllocaOp of type `u8` to return a
    // pointer of type `void *`. This will require a change to the allocaOp
    // verifier.
    auto AllocaAddr = builder.createAlloca(
        getLoc(E->getSourceRange()), builder.getUInt8PtrTy(),
        builder.getUInt8Ty(), "bi_alloca", SuitableAlignmentInBytes, Size);

    // Initialize the allocated buffer if required.
    if (BuiltinID != Builtin::BI__builtin_alloca_uninitialized)
      initializeAlloca(*this, AllocaAddr, Size, SuitableAlignmentInBytes);

    // An alloca will always return a pointer to the alloca (stack) address
    // space. This address space need not be the same as the AST / Language
    // default (e.g. in C / C++ auto vars are in the generic address space). At
    // the AST level this is handled within CreateTempAlloca et al., but for the
    // builtin / dynamic alloca we have to handle it here.
    assert(!cir::MissingFeatures::addressSpace());
    auto AAS = getCIRAllocaAddressSpace();
    auto EAS = builder.getAddrSpaceAttr(
        E->getType()->getPointeeType().getAddressSpace());
    if (EAS != AAS) {
      assert(false && "Non-default address space for alloca NYI");
    }

    // Bitcast the alloca to the expected type.
    return RValue::get(
        builder.createBitcast(AllocaAddr, builder.getVoidPtrTy()));
  }

  case Builtin::BI__builtin_alloca_with_align_uninitialized:
  case Builtin::BI__builtin_alloca_with_align:
    llvm_unreachable("BI__builtin_alloca_with_align like NYI");

  case Builtin::BIbzero:
  case Builtin::BI__builtin_bzero:
    llvm_unreachable("BIbzero like NYI");

  case Builtin::BIbcopy:
  case Builtin::BI__builtin_bcopy:
    llvm_unreachable("BIbcopy like NYI");

  case Builtin::BImemcpy:
  case Builtin::BI__builtin_memcpy:
  case Builtin::BImempcpy:
  case Builtin::BI__builtin_mempcpy: {
    Address Dest = emitPointerWithAlignment(E->getArg(0));
    Address Src = emitPointerWithAlignment(E->getArg(1));
    mlir::Value SizeVal = emitScalarExpr(E->getArg(2));
    emitNonNullArgCheck(RValue::get(Dest.getPointer()), E->getArg(0)->getType(),
                        E->getArg(0)->getExprLoc(), FD, 0);
    emitNonNullArgCheck(RValue::get(Src.getPointer()), E->getArg(1)->getType(),
                        E->getArg(1)->getExprLoc(), FD, 1);
    builder.createMemCpy(getLoc(E->getSourceRange()), Dest.getPointer(),
                         Src.getPointer(), SizeVal);
    if (BuiltinID == Builtin::BImempcpy ||
        BuiltinID == Builtin::BI__builtin_mempcpy)
      llvm_unreachable("mempcpy is NYI");
    else
      return RValue::get(Dest.getPointer());
  }

  case Builtin::BI__builtin_memcpy_inline: {
    Address dest = emitPointerWithAlignment(E->getArg(0));
    Address src = emitPointerWithAlignment(E->getArg(1));
    emitNonNullArgCheck(RValue::get(dest.getPointer()), E->getArg(0)->getType(),
                        E->getArg(0)->getExprLoc(), FD, 0);
    emitNonNullArgCheck(RValue::get(src.getPointer()), E->getArg(1)->getType(),
                        E->getArg(1)->getExprLoc(), FD, 1);
    uint64_t size =
        E->getArg(2)->EvaluateKnownConstInt(getContext()).getZExtValue();
    builder.create<cir::MemCpyInlineOp>(
        getLoc(E->getSourceRange()), dest.getPointer(), src.getPointer(),
        mlir::IntegerAttr::get(mlir::IntegerType::get(builder.getContext(), 64),
                               size));
    // __builtin_memcpy_inline has no return value
    return RValue::get(nullptr);
  }

  case Builtin::BI__builtin_char_memchr:
  case Builtin::BI__builtin_memchr: {
    Address srcPtr = emitPointerWithAlignment(E->getArg(0));
    mlir::Value src =
        builder.createBitcast(srcPtr.getPointer(), builder.getVoidPtrTy());
    mlir::Value pattern = emitScalarExpr(E->getArg(1));
    mlir::Value len = emitScalarExpr(E->getArg(2));
    mlir::Value res =
        builder.create<MemChrOp>(getLoc(E->getExprLoc()), src, pattern, len);
    return RValue::get(res);
  }

  case Builtin::BI__builtin___memcpy_chk: {
    // fold __builtin_memcpy_chk(x, y, cst1, cst2) to memcpy iff cst1<=cst2.
    llvm::APSInt size;
    if (isMemBuiltinOutOfBoundPossible(E->getArg(2), E->getArg(3),
                                       CGM.getASTContext(), size))
      break;
    Address dest = emitPointerWithAlignment(E->getArg(0));
    Address src = emitPointerWithAlignment(E->getArg(1));
    auto loc = getLoc(E->getSourceRange());
    ConstantOp sizeOp = builder.getConstInt(loc, size);
    builder.createMemCpy(loc, dest.getPointer(), src.getPointer(), sizeOp);
    return RValue::get(dest.getPointer());
  }

  case Builtin::BI__builtin_objc_memmove_collectable:
    llvm_unreachable("BI__builtin_objc_memmove_collectable NYI");

  case Builtin::BI__builtin___memmove_chk: {
    // fold __builtin_memcpy_chk(x, y, cst1, cst2) to memcpy iff cst1<=cst2.
    llvm::APSInt size;
    if (isMemBuiltinOutOfBoundPossible(E->getArg(2), E->getArg(3),
                                       CGM.getASTContext(), size))
      break;
    Address Dest = emitPointerWithAlignment(E->getArg(0));
    Address Src = emitPointerWithAlignment(E->getArg(1));
    auto loc = getLoc(E->getSourceRange());
    ConstantOp sizeOp = builder.getConstInt(loc, size);
    builder.createMemMove(loc, Dest.getPointer(), Src.getPointer(), sizeOp);
    return RValue::get(Dest.getPointer());
  }
  case Builtin::BImemmove:
  case Builtin::BI__builtin_memmove: {
    Address Dest = emitPointerWithAlignment(E->getArg(0));
    Address Src = emitPointerWithAlignment(E->getArg(1));
    mlir::Value SizeVal = emitScalarExpr(E->getArg(2));
    emitNonNullArgCheck(RValue::get(Dest.getPointer()), E->getArg(0)->getType(),
                        E->getArg(0)->getExprLoc(), FD, 0);
    emitNonNullArgCheck(RValue::get(Src.getPointer()), E->getArg(1)->getType(),
                        E->getArg(1)->getExprLoc(), FD, 1);
    builder.createMemMove(getLoc(E->getSourceRange()), Dest.getPointer(),
                          Src.getPointer(), SizeVal);
    return RValue::get(Dest.getPointer());
  }
  case Builtin::BImemset:
  case Builtin::BI__builtin_memset: {
    Address Dest = emitPointerWithAlignment(E->getArg(0));
    mlir::Value ByteVal = emitScalarExpr(E->getArg(1));
    mlir::Value SizeVal = emitScalarExpr(E->getArg(2));
    emitNonNullArgCheck(RValue::get(Dest.getPointer()), E->getArg(0)->getType(),
                        E->getArg(0)->getExprLoc(), FD, 0);
    builder.createMemSet(getLoc(E->getSourceRange()), Dest.getPointer(),
                         ByteVal, SizeVal);
    return RValue::get(Dest.getPointer());
  }

  case Builtin::BI__builtin_memset_inline: {
    Address Dest = emitPointerWithAlignment(E->getArg(0));
    mlir::Value ByteVal = emitScalarExpr(E->getArg(1));
    uint64_t size =
        E->getArg(2)->EvaluateKnownConstInt(getContext()).getZExtValue();
    emitNonNullArgCheck(RValue::get(Dest.getPointer()), E->getArg(0)->getType(),
                        E->getArg(0)->getExprLoc(), FD, 0);
    builder.createMemSetInline(
        getLoc(E->getSourceRange()), Dest.getPointer(), ByteVal,
        mlir::IntegerAttr::get(mlir::IntegerType::get(builder.getContext(), 64),
                               size));
    // __builtin_memset_inline has no return value
    return RValue::get(nullptr);
  }
  case Builtin::BI__builtin___memset_chk: {
    // fold __builtin_memset_chk(x, y, cst1, cst2) to memset iff cst1<=cst2.
    llvm::APSInt size;
    if (isMemBuiltinOutOfBoundPossible(E->getArg(2), E->getArg(3),
                                       CGM.getASTContext(), size))
      break;
    Address dest = emitPointerWithAlignment(E->getArg(0));
    mlir::Value byteVal = emitScalarExpr(E->getArg(1));
    auto loc = getLoc(E->getSourceRange());
    ConstantOp sizeOp = builder.getConstInt(loc, size);
    builder.createMemSet(loc, dest.getPointer(), byteVal, sizeOp);
    return RValue::get(dest.getPointer());
  }
  case Builtin::BI__builtin_wmemchr: {
    // The MSVC runtime library does not provide a definition of wmemchr, so we
    // need an inline implementation.
    if (getTarget().getTriple().isOSMSVCRT())
      llvm_unreachable("BI__builtin_wmemchr NYI for OS with MSVC runtime");
    break;
  }
  case Builtin::BI__builtin_wmemcmp:
    llvm_unreachable("BI__builtin_wmemcmp NYI");
  case Builtin::BI__builtin_dwarf_cfa:
    llvm_unreachable("BI__builtin_dwarf_cfa NYI");
  case Builtin::BI__builtin_return_address:
  case Builtin::BI__builtin_frame_address: {
    mlir::Location loc = getLoc(E->getExprLoc());
    mlir::Attribute levelAttr = ConstantEmitter(*this).emitAbstract(
        E->getArg(0), E->getArg(0)->getType());
    uint64_t level = mlir::cast<cir::IntAttr>(levelAttr).getUInt();
    if (BuiltinID == Builtin::BI__builtin_return_address) {
      return RValue::get(builder.create<cir::ReturnAddrOp>(
          loc, builder.getUInt32(level, loc)));
    }
    return RValue::get(
        builder.create<cir::FrameAddrOp>(loc, builder.getUInt32(level, loc)));
  }
  case Builtin::BI_ReturnAddress:
    llvm_unreachable("BI_ReturnAddress NYI");
  case Builtin::BI__builtin_extract_return_addr:
    llvm_unreachable("BI__builtin_extract_return_addr NYI");
  case Builtin::BI__builtin_frob_return_addr:
    llvm_unreachable("BI__builtin_frob_return_addr NYI");
  case Builtin::BI__builtin_dwarf_sp_column:
    llvm_unreachable("BI__builtin_dwarf_sp_column NYI");
  case Builtin::BI__builtin_init_dwarf_reg_size_table:
    llvm_unreachable("BI__builtin_init_dwarf_reg_size_table NYI");
  case Builtin::BI__builtin_eh_return:
    llvm_unreachable("BI__builtin_eh_return NYI");
  case Builtin::BI__builtin_unwind_init:
    llvm_unreachable("BI__builtin_unwind_init NYI");
  case Builtin::BI__builtin_extend_pointer:
    llvm_unreachable("BI__builtin_extend_pointer NYI");
  case Builtin::BI__builtin_setjmp:
    llvm_unreachable("BI__builtin_setjmp NYI");
  case Builtin::BI__builtin_longjmp:
    llvm_unreachable("BI__builtin_longjmp NYI");
  case Builtin::BI__builtin_launder: {
    const clang::Expr *arg = E->getArg(0);
    clang::QualType argTy = arg->getType()->getPointeeType();
    mlir::Value ptr = emitScalarExpr(arg);
    if (typeRequiresBuiltinLaunder(CGM, argTy)) {
      assert(!MissingFeatures::createLaunderInvariantGroup());
      llvm_unreachable(" launder.invariant.group NYI ");
    }
    return RValue::get(ptr);
  }

  case Builtin::BI__sync_fetch_and_add:
  case Builtin::BI__sync_fetch_and_sub:
  case Builtin::BI__sync_fetch_and_or:
  case Builtin::BI__sync_fetch_and_and:
  case Builtin::BI__sync_fetch_and_xor:
  case Builtin::BI__sync_fetch_and_nand:
  case Builtin::BI__sync_add_and_fetch:
  case Builtin::BI__sync_sub_and_fetch:
  case Builtin::BI__sync_and_and_fetch:
  case Builtin::BI__sync_or_and_fetch:
  case Builtin::BI__sync_xor_and_fetch:
  case Builtin::BI__sync_nand_and_fetch:
  case Builtin::BI__sync_val_compare_and_swap:
  case Builtin::BI__sync_bool_compare_and_swap:
  case Builtin::BI__sync_lock_test_and_set:
  case Builtin::BI__sync_lock_release:
  case Builtin::BI__sync_swap:
    llvm_unreachable("Shouldn't make it through sema");

  case Builtin::BI__sync_fetch_and_add_1:
  case Builtin::BI__sync_fetch_and_add_2:
  case Builtin::BI__sync_fetch_and_add_4:
  case Builtin::BI__sync_fetch_and_add_8:
  case Builtin::BI__sync_fetch_and_add_16: {
    return emitBinaryAtomic(*this, cir::AtomicFetchKind::Add, E);
  }

  case Builtin::BI__sync_fetch_and_sub_1:
  case Builtin::BI__sync_fetch_and_sub_2:
  case Builtin::BI__sync_fetch_and_sub_4:
  case Builtin::BI__sync_fetch_and_sub_8:
  case Builtin::BI__sync_fetch_and_sub_16: {
    return emitBinaryAtomic(*this, cir::AtomicFetchKind::Sub, E);
  }

  case Builtin::BI__sync_fetch_and_or_1:
  case Builtin::BI__sync_fetch_and_or_2:
  case Builtin::BI__sync_fetch_and_or_4:
  case Builtin::BI__sync_fetch_and_or_8:
  case Builtin::BI__sync_fetch_and_or_16:
    llvm_unreachable("BI__sync_fetch_and_or NYI");
  case Builtin::BI__sync_fetch_and_and_1:
  case Builtin::BI__sync_fetch_and_and_2:
  case Builtin::BI__sync_fetch_and_and_4:
  case Builtin::BI__sync_fetch_and_and_8:
  case Builtin::BI__sync_fetch_and_and_16:
    llvm_unreachable("BI__sync_fetch_and_and NYI");
  case Builtin::BI__sync_fetch_and_xor_1:
  case Builtin::BI__sync_fetch_and_xor_2:
  case Builtin::BI__sync_fetch_and_xor_4:
  case Builtin::BI__sync_fetch_and_xor_8:
  case Builtin::BI__sync_fetch_and_xor_16:
    llvm_unreachable("BI__sync_fetch_and_xor NYI");
  case Builtin::BI__sync_fetch_and_nand_1:
  case Builtin::BI__sync_fetch_and_nand_2:
  case Builtin::BI__sync_fetch_and_nand_4:
  case Builtin::BI__sync_fetch_and_nand_8:
  case Builtin::BI__sync_fetch_and_nand_16:
    llvm_unreachable("BI__sync_fetch_and_nand NYI");

  // Clang extensions: not overloaded yet.
  case Builtin::BI__sync_fetch_and_min:
    llvm_unreachable("BI__sync_fetch_and_min NYI");
  case Builtin::BI__sync_fetch_and_max:
    llvm_unreachable("BI__sync_fetch_and_max NYI");
  case Builtin::BI__sync_fetch_and_umin:
    llvm_unreachable("BI__sync_fetch_and_umin NYI");
  case Builtin::BI__sync_fetch_and_umax:
    llvm_unreachable("BI__sync_fetch_and_umax NYI");

  case Builtin::BI__sync_add_and_fetch_1:
  case Builtin::BI__sync_add_and_fetch_2:
  case Builtin::BI__sync_add_and_fetch_4:
  case Builtin::BI__sync_add_and_fetch_8:
  case Builtin::BI__sync_add_and_fetch_16:
    return emitBinaryAtomicPost(*this, cir::AtomicFetchKind::Add, E,
                                cir::BinOpKind::Add);

  case Builtin::BI__sync_sub_and_fetch_1:
  case Builtin::BI__sync_sub_and_fetch_2:
  case Builtin::BI__sync_sub_and_fetch_4:
  case Builtin::BI__sync_sub_and_fetch_8:
  case Builtin::BI__sync_sub_and_fetch_16:
    return emitBinaryAtomicPost(*this, cir::AtomicFetchKind::Sub, E,
                                cir::BinOpKind::Sub);

  case Builtin::BI__sync_and_and_fetch_1:
  case Builtin::BI__sync_and_and_fetch_2:
  case Builtin::BI__sync_and_and_fetch_4:
  case Builtin::BI__sync_and_and_fetch_8:
  case Builtin::BI__sync_and_and_fetch_16:
    return emitBinaryAtomicPost(*this, cir::AtomicFetchKind::And, E,
                                cir::BinOpKind::And);

  case Builtin::BI__sync_or_and_fetch_1:
  case Builtin::BI__sync_or_and_fetch_2:
  case Builtin::BI__sync_or_and_fetch_4:
  case Builtin::BI__sync_or_and_fetch_8:
  case Builtin::BI__sync_or_and_fetch_16:
    return emitBinaryAtomicPost(*this, cir::AtomicFetchKind::Or, E,
                                cir::BinOpKind::Or);

  case Builtin::BI__sync_xor_and_fetch_1:
  case Builtin::BI__sync_xor_and_fetch_2:
  case Builtin::BI__sync_xor_and_fetch_4:
  case Builtin::BI__sync_xor_and_fetch_8:
  case Builtin::BI__sync_xor_and_fetch_16:
    return emitBinaryAtomicPost(*this, cir::AtomicFetchKind::Xor, E,
                                cir::BinOpKind::Xor);

  case Builtin::BI__sync_nand_and_fetch_1:
  case Builtin::BI__sync_nand_and_fetch_2:
  case Builtin::BI__sync_nand_and_fetch_4:
  case Builtin::BI__sync_nand_and_fetch_8:
  case Builtin::BI__sync_nand_and_fetch_16:
    return emitBinaryAtomicPost(*this, cir::AtomicFetchKind::Nand, E,
                                cir::BinOpKind::And, true);

  case Builtin::BI__sync_val_compare_and_swap_1:
  case Builtin::BI__sync_val_compare_and_swap_2:
  case Builtin::BI__sync_val_compare_and_swap_4:
  case Builtin::BI__sync_val_compare_and_swap_8:
  case Builtin::BI__sync_val_compare_and_swap_16:
    return RValue::get(MakeAtomicCmpXchgValue(*this, E, false));

  case Builtin::BI__sync_bool_compare_and_swap_1:
  case Builtin::BI__sync_bool_compare_and_swap_2:
  case Builtin::BI__sync_bool_compare_and_swap_4:
  case Builtin::BI__sync_bool_compare_and_swap_8:
  case Builtin::BI__sync_bool_compare_and_swap_16:
    return RValue::get(MakeAtomicCmpXchgValue(*this, E, true));

  case Builtin::BI__sync_swap_1:
  case Builtin::BI__sync_swap_2:
  case Builtin::BI__sync_swap_4:
  case Builtin::BI__sync_swap_8:
  case Builtin::BI__sync_swap_16:
    llvm_unreachable("BI__sync_swap1 like NYI");

  case Builtin::BI__sync_lock_test_and_set_1:
  case Builtin::BI__sync_lock_test_and_set_2:
  case Builtin::BI__sync_lock_test_and_set_4:
  case Builtin::BI__sync_lock_test_and_set_8:
  case Builtin::BI__sync_lock_test_and_set_16:
    llvm_unreachable("BI__sync_lock_test_and_set_1 like NYI");

  case Builtin::BI__sync_lock_release_1:
  case Builtin::BI__sync_lock_release_2:
  case Builtin::BI__sync_lock_release_4:
  case Builtin::BI__sync_lock_release_8:
  case Builtin::BI__sync_lock_release_16:
    llvm_unreachable("BI__sync_lock_release_1 like NYI");

  case Builtin::BI__sync_synchronize:
    llvm_unreachable("BI__sync_synchronize NYI");
  case Builtin::BI__builtin_nontemporal_load:
    llvm_unreachable("BI__builtin_nontemporal_load NYI");
  case Builtin::BI__builtin_nontemporal_store:
    llvm_unreachable("BI__builtin_nontemporal_store NYI");
  case Builtin::BI__c11_atomic_is_lock_free:
    llvm_unreachable("BI__c11_atomic_is_lock_free NYI");
  case Builtin::BI__atomic_is_lock_free:
    llvm_unreachable("BI__atomic_is_lock_free NYI");
  case Builtin::BI__atomic_test_and_set:
    llvm_unreachable("BI__atomic_test_and_set NYI");
  case Builtin::BI__atomic_clear:
    llvm_unreachable("BI__atomic_clear NYI");

  case Builtin::BI__atomic_thread_fence:
    return RValue::get(
        makeAtomicFenceValue(*this, E, cir::MemScopeKind::MemScope_System));
  case Builtin::BI__atomic_signal_fence:
    return RValue::get(makeAtomicFenceValue(
        *this, E, cir::MemScopeKind::MemScope_SingleThread));
  case Builtin::BI__c11_atomic_thread_fence:
  case Builtin::BI__c11_atomic_signal_fence:
    llvm_unreachable("BI__c11_atomic_thread_fence like NYI");

  case Builtin::BI__builtin_signbit:
  case Builtin::BI__builtin_signbitf:
  case Builtin::BI__builtin_signbitl: {
    auto loc = getLoc(E->getBeginLoc());
    return RValue::get(builder.createZExtOrBitCast(
        loc, emitSignBit(loc, *this, emitScalarExpr(E->getArg(0))),
        convertType(E->getType())));
  }

  case Builtin::BI__warn_memset_zero_len:
    llvm_unreachable("BI__warn_memset_zero_len NYI");
  case Builtin::BI__annotation:
    llvm_unreachable("BI__annotation NYI");
  case Builtin::BI__builtin_annotation:
    llvm_unreachable("BI__builtin_annotation NYI");
  case Builtin::BI__builtin_addcb:
  case Builtin::BI__builtin_addcs:
  case Builtin::BI__builtin_addc:
  case Builtin::BI__builtin_addcl:
  case Builtin::BI__builtin_addcll:
  case Builtin::BI__builtin_subcb:
  case Builtin::BI__builtin_subcs:
  case Builtin::BI__builtin_subc:
  case Builtin::BI__builtin_subcl:
  case Builtin::BI__builtin_subcll:
    llvm_unreachable("BI__builtin_addcb like NYI");

  case Builtin::BI__builtin_add_overflow:
  case Builtin::BI__builtin_sub_overflow:
  case Builtin::BI__builtin_mul_overflow: {
    const clang::Expr *LeftArg = E->getArg(0);
    const clang::Expr *RightArg = E->getArg(1);
    const clang::Expr *ResultArg = E->getArg(2);

    clang::QualType ResultQTy =
        ResultArg->getType()->castAs<clang::PointerType>()->getPointeeType();

    WidthAndSignedness LeftInfo =
        getIntegerWidthAndSignedness(CGM.getASTContext(), LeftArg->getType());
    WidthAndSignedness RightInfo =
        getIntegerWidthAndSignedness(CGM.getASTContext(), RightArg->getType());
    WidthAndSignedness ResultInfo =
        getIntegerWidthAndSignedness(CGM.getASTContext(), ResultQTy);

    // Note we compute the encompassing type with the consideration to the
    // result type, so later in LLVM lowering we don't get redundant integral
    // extension casts.
    WidthAndSignedness EncompassingInfo =
        EncompassingIntegerType({LeftInfo, RightInfo, ResultInfo});

    auto EncompassingCIRTy = cir::IntType::get(
        &getMLIRContext(), EncompassingInfo.Width, EncompassingInfo.Signed);
    auto ResultCIRTy = mlir::cast<cir::IntType>(CGM.convertType(ResultQTy));

    mlir::Value Left = emitScalarExpr(LeftArg);
    mlir::Value Right = emitScalarExpr(RightArg);
    Address ResultPtr = emitPointerWithAlignment(ResultArg);

    // Extend each operand to the encompassing type, if necessary.
    if (Left.getType() != EncompassingCIRTy)
      Left =
          builder.createCast(cir::CastKind::integral, Left, EncompassingCIRTy);
    if (Right.getType() != EncompassingCIRTy)
      Right =
          builder.createCast(cir::CastKind::integral, Right, EncompassingCIRTy);

    // Perform the operation on the extended values.
    cir::BinOpOverflowKind OpKind;
    switch (BuiltinID) {
    default:
      llvm_unreachable("Unknown overflow builtin id.");
    case Builtin::BI__builtin_add_overflow:
      OpKind = cir::BinOpOverflowKind::Add;
      break;
    case Builtin::BI__builtin_sub_overflow:
      OpKind = cir::BinOpOverflowKind::Sub;
      break;
    case Builtin::BI__builtin_mul_overflow:
      OpKind = cir::BinOpOverflowKind::Mul;
      break;
    }

    auto Loc = getLoc(E->getSourceRange());
    auto ArithResult =
        builder.createBinOpOverflowOp(Loc, ResultCIRTy, OpKind, Left, Right);

    // Here is a slight difference from the original clang CodeGen:
    //   - In the original clang CodeGen, the checked arithmetic result is
    //     first computed as a value of the encompassing type, and then it is
    //     truncated to the actual result type with a second overflow checking.
    //   - In CIRGen, the checked arithmetic operation directly produce the
    //     checked arithmetic result in its expected type.
    //
    // So we don't need a truncation and a second overflow checking here.

    // Finally, store the result using the pointer.
    bool isVolatile =
        ResultArg->getType()->getPointeeType().isVolatileQualified();
    builder.createStore(Loc, emitToMemory(ArithResult.result, ResultQTy),
                        ResultPtr, isVolatile);

    return RValue::get(ArithResult.overflow);
  }

  case Builtin::BI__builtin_uadd_overflow:
  case Builtin::BI__builtin_uaddl_overflow:
  case Builtin::BI__builtin_uaddll_overflow:
  case Builtin::BI__builtin_usub_overflow:
  case Builtin::BI__builtin_usubl_overflow:
  case Builtin::BI__builtin_usubll_overflow:
  case Builtin::BI__builtin_umul_overflow:
  case Builtin::BI__builtin_umull_overflow:
  case Builtin::BI__builtin_umulll_overflow:
  case Builtin::BI__builtin_sadd_overflow:
  case Builtin::BI__builtin_saddl_overflow:
  case Builtin::BI__builtin_saddll_overflow:
  case Builtin::BI__builtin_ssub_overflow:
  case Builtin::BI__builtin_ssubl_overflow:
  case Builtin::BI__builtin_ssubll_overflow:
  case Builtin::BI__builtin_smul_overflow:
  case Builtin::BI__builtin_smull_overflow:
  case Builtin::BI__builtin_smulll_overflow: {
    // Scalarize our inputs.
    mlir::Value X = emitScalarExpr(E->getArg(0));
    mlir::Value Y = emitScalarExpr(E->getArg(1));

    const clang::Expr *ResultArg = E->getArg(2);
    Address ResultPtr = emitPointerWithAlignment(ResultArg);

    // Decide which of the arithmetic operation we are lowering to:
    cir::BinOpOverflowKind ArithKind;
    switch (BuiltinID) {
    default:
      llvm_unreachable("Unknown overflow builtin id.");
    case Builtin::BI__builtin_uadd_overflow:
    case Builtin::BI__builtin_uaddl_overflow:
    case Builtin::BI__builtin_uaddll_overflow:
    case Builtin::BI__builtin_sadd_overflow:
    case Builtin::BI__builtin_saddl_overflow:
    case Builtin::BI__builtin_saddll_overflow:
      ArithKind = cir::BinOpOverflowKind::Add;
      break;
    case Builtin::BI__builtin_usub_overflow:
    case Builtin::BI__builtin_usubl_overflow:
    case Builtin::BI__builtin_usubll_overflow:
    case Builtin::BI__builtin_ssub_overflow:
    case Builtin::BI__builtin_ssubl_overflow:
    case Builtin::BI__builtin_ssubll_overflow:
      ArithKind = cir::BinOpOverflowKind::Sub;
      break;
    case Builtin::BI__builtin_umul_overflow:
    case Builtin::BI__builtin_umull_overflow:
    case Builtin::BI__builtin_umulll_overflow:
    case Builtin::BI__builtin_smul_overflow:
    case Builtin::BI__builtin_smull_overflow:
    case Builtin::BI__builtin_smulll_overflow:
      ArithKind = cir::BinOpOverflowKind::Mul;
      break;
    }

    clang::QualType ResultQTy =
        ResultArg->getType()->castAs<clang::PointerType>()->getPointeeType();
    auto ResultCIRTy = mlir::cast<cir::IntType>(CGM.convertType(ResultQTy));

    auto Loc = getLoc(E->getSourceRange());
    auto ArithResult =
        builder.createBinOpOverflowOp(Loc, ResultCIRTy, ArithKind, X, Y);

    bool isVolatile =
        ResultArg->getType()->getPointeeType().isVolatileQualified();
    builder.createStore(Loc, emitToMemory(ArithResult.result, ResultQTy),
                        ResultPtr, isVolatile);

    return RValue::get(ArithResult.overflow);
  }

  case Builtin::BIaddressof:
  case Builtin::BI__addressof:
  case Builtin::BI__builtin_addressof:
    return RValue::get(emitLValue(E->getArg(0)).getPointer());
  case Builtin::BI__builtin_function_start:
    llvm_unreachable("BI__builtin_function_start NYI");
  case Builtin::BI__builtin_operator_new:
    return emitBuiltinNewDeleteCall(
        E->getCallee()->getType()->castAs<FunctionProtoType>(), E, false);
  case Builtin::BI__builtin_operator_delete:
    emitBuiltinNewDeleteCall(
        E->getCallee()->getType()->castAs<FunctionProtoType>(), E, true);
    return RValue::get(nullptr);
  case Builtin::BI__builtin_is_aligned:
    llvm_unreachable("BI__builtin_is_aligned NYI");
  case Builtin::BI__builtin_align_up:
    llvm_unreachable("BI__builtin_align_up NYI");
  case Builtin::BI__builtin_align_down:
    llvm_unreachable("BI__builtin_align_down NYI");

  case Builtin::BI__noop:
    // __noop always evaluates to an integer literal zero.
    llvm_unreachable("BI__noop NYI");
  case Builtin::BI__builtin_call_with_static_chain:
    llvm_unreachable("BI__builtin_call_with_static_chain NYI");
  case Builtin::BI_InterlockedExchange8:
  case Builtin::BI_InterlockedExchange16:
  case Builtin::BI_InterlockedExchange:
  case Builtin::BI_InterlockedExchangePointer:
    llvm_unreachable("BI_InterlockedExchange8 like NYI");
  case Builtin::BI_InterlockedCompareExchangePointer:
  case Builtin::BI_InterlockedCompareExchangePointer_nf:
    llvm_unreachable("BI_InterlockedCompareExchangePointer like NYI");
  case Builtin::BI_InterlockedCompareExchange8:
  case Builtin::BI_InterlockedCompareExchange16:
  case Builtin::BI_InterlockedCompareExchange:
  case Builtin::BI_InterlockedCompareExchange64:
    llvm_unreachable("BI_InterlockedCompareExchange8 like NYI");
  case Builtin::BI_InterlockedIncrement16:
  case Builtin::BI_InterlockedIncrement:
    llvm_unreachable("BI_InterlockedIncrement16 like NYI");
  case Builtin::BI_InterlockedDecrement16:
  case Builtin::BI_InterlockedDecrement:
    llvm_unreachable("BI_InterlockedDecrement16 like NYI");
  case Builtin::BI_InterlockedAnd8:
  case Builtin::BI_InterlockedAnd16:
  case Builtin::BI_InterlockedAnd:
    llvm_unreachable("BI_InterlockedAnd8 like NYI");
  case Builtin::BI_InterlockedExchangeAdd8:
  case Builtin::BI_InterlockedExchangeAdd16:
  case Builtin::BI_InterlockedExchangeAdd:
    llvm_unreachable("BI_InterlockedExchangeAdd8 like NYI");
  case Builtin::BI_InterlockedExchangeSub8:
  case Builtin::BI_InterlockedExchangeSub16:
  case Builtin::BI_InterlockedExchangeSub:
    llvm_unreachable("BI_InterlockedExchangeSub8 like NYI");
  case Builtin::BI_InterlockedOr8:
  case Builtin::BI_InterlockedOr16:
  case Builtin::BI_InterlockedOr:
    llvm_unreachable("BI_InterlockedOr8 like NYI");
  case Builtin::BI_InterlockedXor8:
  case Builtin::BI_InterlockedXor16:
  case Builtin::BI_InterlockedXor:
    llvm_unreachable("BI_InterlockedXor8 like NYI");

  case Builtin::BI_bittest64:
  case Builtin::BI_bittest:
  case Builtin::BI_bittestandcomplement64:
  case Builtin::BI_bittestandcomplement:
  case Builtin::BI_bittestandreset64:
  case Builtin::BI_bittestandreset:
  case Builtin::BI_bittestandset64:
  case Builtin::BI_bittestandset:
  case Builtin::BI_interlockedbittestandreset:
  case Builtin::BI_interlockedbittestandreset64:
  case Builtin::BI_interlockedbittestandset64:
  case Builtin::BI_interlockedbittestandset:
  case Builtin::BI_interlockedbittestandset_acq:
  case Builtin::BI_interlockedbittestandset_rel:
  case Builtin::BI_interlockedbittestandset_nf:
  case Builtin::BI_interlockedbittestandreset_acq:
  case Builtin::BI_interlockedbittestandreset_rel:
  case Builtin::BI_interlockedbittestandreset_nf:
    llvm_unreachable("BI_bittest64 like NYI");

  // These builtins exist to emit regular volatile loads and stores not
  // affected by the -fms-volatile setting.
  case Builtin::BI__iso_volatile_load8:
  case Builtin::BI__iso_volatile_load16:
  case Builtin::BI__iso_volatile_load32:
  case Builtin::BI__iso_volatile_load64:
    llvm_unreachable("BI__iso_volatile_load8 like NYI");
  case Builtin::BI__iso_volatile_store8:
  case Builtin::BI__iso_volatile_store16:
  case Builtin::BI__iso_volatile_store32:
  case Builtin::BI__iso_volatile_store64:
    llvm_unreachable("BI__iso_volatile_store8 like NYI");

  case Builtin::BI__builtin_ptrauth_sign_constant:
    llvm_unreachable("BI__builtin_ptrauth_sign_constant NYI");

  case Builtin::BI__builtin_ptrauth_auth:
  case Builtin::BI__builtin_ptrauth_auth_and_resign:
  case Builtin::BI__builtin_ptrauth_blend_discriminator:
  case Builtin::BI__builtin_ptrauth_sign_generic_data:
  case Builtin::BI__builtin_ptrauth_sign_unauthenticated:
  case Builtin::BI__builtin_ptrauth_strip:
    llvm_unreachable("BI__builtin_ptrauth_auth like NYI");

  case Builtin::BI__exception_code:
  case Builtin::BI_exception_code:
    llvm_unreachable("BI__exception_code like NYI");
  case Builtin::BI__exception_info:
  case Builtin::BI_exception_info:
    llvm_unreachable("BI__exception_info like NYI");
  case Builtin::BI__abnormal_termination:
  case Builtin::BI_abnormal_termination:
    llvm_unreachable("BI__abnormal_termination like NYI");
  case Builtin::BI_setjmpex:
    llvm_unreachable("BI_setjmpex NYI");
    break;
  case Builtin::BI_setjmp:
    llvm_unreachable("BI_setjmp NYI");
    break;

  // C++ std:: builtins.
  case Builtin::BImove:
  case Builtin::BImove_if_noexcept:
  case Builtin::BIforward:
  case Builtin::BIas_const:
    return RValue::get(emitLValue(E->getArg(0)).getPointer());
  case Builtin::BIforward_like:
    llvm_unreachable("BIforward_like NYI");
  case Builtin::BI__GetExceptionInfo:
    llvm_unreachable("BI__GetExceptionInfo NYI");

  case Builtin::BI__fastfail:
    llvm_unreachable("BI__fastfail NYI");

  case Builtin::BI__builtin_coro_id:
  case Builtin::BI__builtin_coro_promise:
  case Builtin::BI__builtin_coro_resume:
  case Builtin::BI__builtin_coro_noop:
  case Builtin::BI__builtin_coro_destroy:
  case Builtin::BI__builtin_coro_done:
  case Builtin::BI__builtin_coro_alloc:
  case Builtin::BI__builtin_coro_begin:
  case Builtin::BI__builtin_coro_end:
  case Builtin::BI__builtin_coro_suspend:
  case Builtin::BI__builtin_coro_align:
    llvm_unreachable("BI__builtin_coro_id like NYI");

  case Builtin::BI__builtin_coro_frame: {
    return emitCoroutineFrame();
  }
  case Builtin::BI__builtin_coro_free:
  case Builtin::BI__builtin_coro_size: {
    GlobalDecl gd{FD};
    mlir::Type ty = CGM.getTypes().GetFunctionType(
        CGM.getTypes().arrangeGlobalDeclaration(GD));
    const auto *ND = cast<NamedDecl>(GD.getDecl());
    auto fnOp =
        CGM.GetOrCreateCIRFunction(ND->getName(), ty, gd, /*ForVTable=*/false,
                                   /*DontDefer=*/false);
    fnOp.setBuiltinAttr(mlir::UnitAttr::get(&getMLIRContext()));
    return emitCall(E->getCallee()->getType(), CIRGenCallee::forDirect(fnOp), E,
                    ReturnValue);
  }

  case Builtin::BIread_pipe:
  case Builtin::BIwrite_pipe:
    llvm_unreachable("BIread_pipe and BIwrite_pipe NYI");

  // OpenCL v2.0 s6.13.16 ,s9.17.3.5 - Built-in pipe reserve read and write
  // functions
  case Builtin::BIreserve_read_pipe:
  case Builtin::BIreserve_write_pipe:
  case Builtin::BIwork_group_reserve_read_pipe:
  case Builtin::BIwork_group_reserve_write_pipe:
  case Builtin::BIsub_group_reserve_read_pipe:
  case Builtin::BIsub_group_reserve_write_pipe:
    llvm_unreachable("BIreserve_read_pipe like NYI");

  // OpenCL v2.0 s6.13.16, s9.17.3.5 - Built-in pipe commit read and write
  // functions
  case Builtin::BIcommit_read_pipe:
  case Builtin::BIcommit_write_pipe:
  case Builtin::BIwork_group_commit_read_pipe:
  case Builtin::BIwork_group_commit_write_pipe:
  case Builtin::BIsub_group_commit_read_pipe:
  case Builtin::BIsub_group_commit_write_pipe:
    llvm_unreachable("BIcommit_read_pipe like NYI");
  // OpenCL v2.0 s6.13.16.4 Built-in pipe query functions
  case Builtin::BIget_pipe_num_packets:
  case Builtin::BIget_pipe_max_packets:
    llvm_unreachable("BIget_pipe_num_packets like NYI");

  // OpenCL v2.0 s6.13.9 - Address space qualifier functions.
  case Builtin::BIto_global:
  case Builtin::BIto_local:
  case Builtin::BIto_private:
    llvm_unreachable("Builtin::BIto_global like NYI");

  // OpenCL v2.0, s6.13.17 - Enqueue kernel function.
  // Table 6.13.17.1 specifies four overload forms of enqueue_kernel.
  // The code below expands the builtin call to a call to one of the following
  // functions that an OpenCL runtime library will have to provide:
  //   __enqueue_kernel_basic
  //   __enqueue_kernel_varargs
  //   __enqueue_kernel_basic_events
  //   __enqueue_kernel_events_varargs
  case Builtin::BIenqueue_kernel:
    llvm_unreachable("BIenqueue_kernel NYI");
  // OpenCL v2.0 s6.13.17.6 - Kernel query functions need bitcast of block
  // parameter.
  case Builtin::BIget_kernel_work_group_size:
    llvm_unreachable("BIget_kernel_work_group_size NYI");
  case Builtin::BIget_kernel_preferred_work_group_size_multiple:
    llvm_unreachable("BIget_kernel_preferred_work_group_size_multiple NYI");

  case Builtin::BIget_kernel_max_sub_group_size_for_ndrange:
  case Builtin::BIget_kernel_sub_group_count_for_ndrange:
    llvm_unreachable("BIget_kernel_max_sub_group_size_for_ndrange like NYI");

  case Builtin::BI__builtin_store_half:
  case Builtin::BI__builtin_store_halff:
    llvm_unreachable("BI__builtin_store_half like NYI");
  case Builtin::BI__builtin_load_half:
    llvm_unreachable("BI__builtin_load_half NYI");
  case Builtin::BI__builtin_load_halff:
    llvm_unreachable("BI__builtin_load_halff NYI");

  case Builtin::BI__builtin_printf:
    llvm_unreachable("BI__builtin_printf NYI");
  case Builtin::BIprintf:
    if (getTarget().getTriple().isNVPTX() ||
        getTarget().getTriple().isAMDGCN()) {
      llvm_unreachable("BIprintf NYI");
    }
    break;

  case Builtin::BI__builtin_canonicalize:
  case Builtin::BI__builtin_canonicalizef:
  case Builtin::BI__builtin_canonicalizef16:
  case Builtin::BI__builtin_canonicalizel:
    llvm_unreachable("BI__builtin_canonicalize like NYI");

  case Builtin::BI__builtin_thread_pointer:
    llvm_unreachable("BI__builtin_thread_pointer NYI");
  case Builtin::BI__builtin_os_log_format:
    llvm_unreachable("BI__builtin_os_log_format NYI");
  case Builtin::BI__xray_customevent:
    llvm_unreachable("BI__xray_customevent NYI");
  case Builtin::BI__xray_typedevent:
    llvm_unreachable("BI__xray_typedevent NYI");

  case Builtin::BI__builtin_ms_va_start:
  case Builtin::BI__builtin_ms_va_end:
    llvm_unreachable("BI__builtin_ms_va_start like NYI");

  case Builtin::BI__builtin_ms_va_copy:
    llvm_unreachable("BI__builtin_ms_va_copy NYI");
  case Builtin::BI__builtin_get_device_side_mangled_name:
    llvm_unreachable("BI__builtin_get_device_side_mangled_name NYI");

  // From https://clang.llvm.org/docs/LanguageExtensions.html#builtin-isfpclass
  // :
  //
  //  The `__builtin_isfpclass()` builtin is a generalization of functions
  //  isnan, isinf, isfinite and some others defined by the C standard. It tests
  //  if the floating-point value, specified by the first argument, falls into
  //  any of data classes, specified by the second argument.
  case Builtin::BI__builtin_isnan: {
    CIRGenFunction::CIRGenFPOptionsRAII FPOptsRAII(*this, E);
    mlir::Value V = emitScalarExpr(E->getArg(0));
    if (mlir::Value Result = tryUseTestFPKind(*this, BuiltinID, V))
      return RValue::get(Result);
    mlir::Location Loc = getLoc(E->getBeginLoc());
    // FIXME: We should use builder.createZExt once createZExt is available.
    return RValue::get(builder.createZExtOrBitCast(
        Loc, builder.createIsFPClass(Loc, V, FPClassTest::fcNan),
        convertType(E->getType())));
  }

  case Builtin::BI__builtin_issignaling: {
    CIRGenFunction::CIRGenFPOptionsRAII FPOptsRAII(*this, E);
    mlir::Value V = emitScalarExpr(E->getArg(0));
    mlir::Location Loc = getLoc(E->getBeginLoc());
    // FIXME: We should use builder.createZExt once createZExt is available.
    return RValue::get(builder.createZExtOrBitCast(
        Loc, builder.createIsFPClass(Loc, V, FPClassTest::fcSNan),
        convertType(E->getType())));
  }

  case Builtin::BI__builtin_isinf: {
    CIRGenFunction::CIRGenFPOptionsRAII FPOptsRAII(*this, E);
    mlir::Value V = emitScalarExpr(E->getArg(0));
    if (mlir::Value Result = tryUseTestFPKind(*this, BuiltinID, V))
      return RValue::get(Result);
    mlir::Location Loc = getLoc(E->getBeginLoc());
    // FIXME: We should use builder.createZExt once createZExt is available.
    return RValue::get(builder.createZExtOrBitCast(
        Loc, builder.createIsFPClass(Loc, V, FPClassTest::fcInf),
        convertType(E->getType())));
  }

  case Builtin::BIfinite:
  case Builtin::BI__finite:
  case Builtin::BIfinitef:
  case Builtin::BI__finitef:
  case Builtin::BIfinitel:
  case Builtin::BI__finitel:
  case Builtin::BI__builtin_isfinite: {
    CIRGenFunction::CIRGenFPOptionsRAII FPOptsRAII(*this, E);
    mlir::Value V = emitScalarExpr(E->getArg(0));
    if (mlir::Value Result = tryUseTestFPKind(*this, BuiltinID, V))
      return RValue::get(Result);
    mlir::Location Loc = getLoc(E->getBeginLoc());
    // FIXME: We should use builder.createZExt once createZExt is available.
    return RValue::get(builder.createZExtOrBitCast(
        Loc, builder.createIsFPClass(Loc, V, FPClassTest::fcFinite),
        convertType(E->getType())));
  }

  case Builtin::BI__builtin_isnormal: {
    CIRGenFunction::CIRGenFPOptionsRAII FPOptsRAII(*this, E);
    mlir::Value V = emitScalarExpr(E->getArg(0));
    mlir::Location Loc = getLoc(E->getBeginLoc());
    // FIXME: We should use builder.createZExt once createZExt is available.
    return RValue::get(builder.createZExtOrBitCast(
        Loc, builder.createIsFPClass(Loc, V, FPClassTest::fcNormal),
        convertType(E->getType())));
  }

  case Builtin::BI__builtin_issubnormal: {
    CIRGenFunction::CIRGenFPOptionsRAII FPOptsRAII(*this, E);
    mlir::Value V = emitScalarExpr(E->getArg(0));
    mlir::Location Loc = getLoc(E->getBeginLoc());
    // FIXME: We should use builder.createZExt once createZExt is available.
    return RValue::get(builder.createZExtOrBitCast(
        Loc, builder.createIsFPClass(Loc, V, FPClassTest::fcSubnormal),
        convertType(E->getType())));
  }

  case Builtin::BI__builtin_iszero: {
    CIRGenFunction::CIRGenFPOptionsRAII FPOptsRAII(*this, E);
    mlir::Value V = emitScalarExpr(E->getArg(0));
    mlir::Location Loc = getLoc(E->getBeginLoc());
    // FIXME: We should use builder.createZExt once createZExt is available.
    return RValue::get(builder.createZExtOrBitCast(
        Loc, builder.createIsFPClass(Loc, V, FPClassTest::fcZero),
        convertType(E->getType())));
  }

  case Builtin::BI__builtin_isfpclass: {
    Expr::EvalResult Result;
    if (!E->getArg(1)->EvaluateAsInt(Result, CGM.getASTContext()))
      break;

    CIRGenFunction::CIRGenFPOptionsRAII FPOptsRAII(*this, E);
    mlir::Value V = emitScalarExpr(E->getArg(0));
    uint64_t Test = Result.Val.getInt().getLimitedValue();
    mlir::Location Loc = getLoc(E->getBeginLoc());

    // FIXME: We should use builder.createZExt once createZExt is available.
    return RValue::get(builder.createZExtOrBitCast(
        Loc, builder.createIsFPClass(Loc, V, Test), convertType(E->getType())));
  }
  }

  // If this is an alias for a lib function (e.g. __builtin_sin), emit
  // the call using the normal call path, but using the unmangled
  // version of the function name.
  if (getContext().BuiltinInfo.isLibFunction(BuiltinID))
    return emitLibraryCall(*this, FD, E,
                           CGM.getBuiltinLibFunction(FD, BuiltinID));

  // If this is a predefined lib function (e.g. malloc), emit the call
  // using exactly the normal call path.
  if (getContext().BuiltinInfo.isPredefinedLibFunction(BuiltinID))
    return emitLibraryCall(*this, FD, E,
                           emitScalarExpr(E->getCallee()).getDefiningOp());

  // Check that a call to a target specific builtin has the correct target
  // features.
  // This is down here to avoid non-target specific builtins, however, if
  // generic builtins start to require generic target features then we
  // can move this up to the beginning of the function.
  //   checkTargetFeatures(E, FD);

  if (unsigned VectorWidth =
          getContext().BuiltinInfo.getRequiredVectorWidth(BuiltinID))
    llvm_unreachable("NYI");

  // See if we have a target specific intrinsic.
  auto Name = getContext().BuiltinInfo.getName(BuiltinID).str();
  Intrinsic::ID IntrinsicID = Intrinsic::not_intrinsic;
  StringRef Prefix =
      llvm::Triple::getArchTypePrefix(getTarget().getTriple().getArch());
  if (!Prefix.empty()) {
    IntrinsicID = Intrinsic::getIntrinsicForClangBuiltin(Prefix.data(), Name);
    // NOTE we don't need to perform a compatibility flag check here since the
    // intrinsics are declared in Builtins*.def via LANGBUILTIN which filter the
    // MS builtins via ALL_MS_LANGUAGES and are filtered earlier.
    if (IntrinsicID == Intrinsic::not_intrinsic)
      IntrinsicID = Intrinsic::getIntrinsicForMSBuiltin(Prefix.data(), Name);
  }

  if (IntrinsicID != Intrinsic::not_intrinsic) {
    llvm_unreachable("NYI");
  }

  // Some target-specific builtins can have aggregate return values, e.g.
  // __builtin_arm_mve_vld2q_u32. So if the result is an aggregate, force
  // ReturnValue to be non-null, so that the target-specific emission code can
  // always just emit into it.
  cir::TypeEvaluationKind EvalKind = getEvaluationKind(E->getType());
  if (EvalKind == cir::TEK_Aggregate && ReturnValue.isNull()) {
    llvm_unreachable("NYI");
  }

  // Now see if we can emit a target-specific builtin.
  if (auto V = emitTargetBuiltinExpr(BuiltinID, E, ReturnValue)) {
    switch (EvalKind) {
    case cir::TEK_Scalar:
      if (mlir::isa<cir::VoidType>(V.getType()))
        return RValue::get(nullptr);
      return RValue::get(V);
    case cir::TEK_Aggregate:
      llvm_unreachable("NYI");
    case cir::TEK_Complex:
      llvm_unreachable("No current target builtin returns complex");
    }
    llvm_unreachable("Bad evaluation kind in EmitBuiltinExpr");
  }

  CGM.ErrorUnsupported(E, "builtin function");

  // Unknown builtin, for now just dump it out and return undef.
  return GetUndefRValue(E->getType());
}

mlir::Value CIRGenFunction::emitCheckedArgForBuiltin(const Expr *E,
                                                     BuiltinCheckKind Kind) {
  assert((Kind == BCK_CLZPassedZero || Kind == BCK_CTZPassedZero) &&
         "Unsupported builtin check kind");

  auto value = emitScalarExpr(E);
  if (!SanOpts.has(SanitizerKind::Builtin))
    return value;

  assert(!cir::MissingFeatures::sanitizerBuiltin());
  llvm_unreachable("NYI");
}

mlir::Value CIRGenFunction::emitCheckedArgForAssume(const Expr *E) {
  mlir::Value argValue = evaluateExprAsBool(E);
  if (!SanOpts.has(SanitizerKind::Builtin))
    return argValue;

  assert(!MissingFeatures::sanitizerBuiltin());
  llvm_unreachable("NYI");
}

static mlir::Value emitTargetArchBuiltinExpr(CIRGenFunction *CGF,
                                             unsigned BuiltinID,
                                             const CallExpr *E,
                                             ReturnValueSlot ReturnValue,
                                             llvm::Triple::ArchType Arch) {
  // When compiling in HipStdPar mode we have to be conservative in rejecting
  // target specific features in the FE, and defer the possible error to the
  // AcceleratorCodeSelection pass, wherein iff an unsupported target builtin is
  // referenced by an accelerator executable function, we emit an error.
  // Returning nullptr here leads to the builtin being handled in
  // EmitStdParUnsupportedBuiltin.
  if (CGF->getLangOpts().HIPStdPar && CGF->getLangOpts().CUDAIsDevice &&
      Arch != CGF->getTarget().getTriple().getArch())
    return nullptr;

  switch (Arch) {
  case llvm::Triple::arm:
  case llvm::Triple::armeb:
  case llvm::Triple::thumb:
  case llvm::Triple::thumbeb:
    llvm_unreachable("NYI");
  case llvm::Triple::aarch64:
  case llvm::Triple::aarch64_32:
  case llvm::Triple::aarch64_be:
    return CGF->emitAArch64BuiltinExpr(BuiltinID, E, ReturnValue, Arch);
  case llvm::Triple::bpfeb:
  case llvm::Triple::bpfel:
    llvm_unreachable("NYI");
  case llvm::Triple::x86:
  case llvm::Triple::x86_64:
    return CGF->emitX86BuiltinExpr(BuiltinID, E);
  case llvm::Triple::ppc:
  case llvm::Triple::ppcle:
  case llvm::Triple::ppc64:
  case llvm::Triple::ppc64le:
    llvm_unreachable("NYI");
  case llvm::Triple::r600:
  case llvm::Triple::amdgcn:
    llvm_unreachable("NYI");
  case llvm::Triple::systemz:
    llvm_unreachable("NYI");
  case llvm::Triple::nvptx:
  case llvm::Triple::nvptx64:
    llvm_unreachable("NYI");
  case llvm::Triple::wasm32:
  case llvm::Triple::wasm64:
    llvm_unreachable("NYI");
  case llvm::Triple::hexagon:
    llvm_unreachable("NYI");
  case llvm::Triple::riscv32:
  case llvm::Triple::riscv64:
    llvm_unreachable("NYI");
  default:
    return {};
  }
}

mlir::Value CIRGenFunction::emitTargetBuiltinExpr(unsigned BuiltinID,
                                                  const CallExpr *E,
                                                  ReturnValueSlot ReturnValue) {
  if (getContext().BuiltinInfo.isAuxBuiltinID(BuiltinID)) {
    assert(getContext().getAuxTargetInfo() && "Missing aux target info");
    return emitTargetArchBuiltinExpr(
        this, getContext().BuiltinInfo.getAuxBuiltinID(BuiltinID), E,
        ReturnValue, getContext().getAuxTargetInfo()->getTriple().getArch());
  }

  return emitTargetArchBuiltinExpr(this, BuiltinID, E, ReturnValue,
                                   getTarget().getTriple().getArch());
}

mlir::Value CIRGenFunction::emitScalarOrConstFoldImmArg(unsigned ICEArguments,
                                                        unsigned Idx,
                                                        const CallExpr *E) {
  mlir::Value Arg = {};
  if ((ICEArguments & (1 << Idx)) == 0) {
    Arg = emitScalarExpr(E->getArg(Idx));
  } else {
    // If this is required to be a constant, constant fold it so that we
    // know that the generated intrinsic gets a ConstantInt.
    std::optional<llvm::APSInt> Result =
        E->getArg(Idx)->getIntegerConstantExpr(getContext());
    assert(Result && "Expected argument to be a constant");
    Arg = builder.getConstInt(getLoc(E->getSourceRange()), *Result);
  }
  return Arg;
}

void CIRGenFunction::emitVAStartEnd(mlir::Value ArgValue, bool IsStart) {
  // LLVM codegen casts to *i8, no real gain on doing this for CIRGen this
  // early, defer to LLVM lowering.
  if (IsStart)
    builder.create<cir::VAStartOp>(ArgValue.getLoc(), ArgValue);
  else
    builder.create<cir::VAEndOp>(ArgValue.getLoc(), ArgValue);
}

/// Checks if using the result of __builtin_object_size(p, @p From) in place of
/// __builtin_object_size(p, @p To) is correct
static bool areBOSTypesCompatible(int From, int To) {
  // Note: Our __builtin_object_size implementation currently treats Type=0 and
  // Type=2 identically. Encoding this implementation detail here may make
  // improving __builtin_object_size difficult in the future, so it's omitted.
  return From == To || (From == 0 && To == 1) || (From == 3 && To == 2);
}

/// Returns a Value corresponding to the size of the given expression.
/// This Value may be either of the following:
///
///   - Reference an argument if `pass_object_size` is used.
///   - A call to a `cir.objsize`.
///
/// EmittedE is the result of emitting `E` as a scalar expr. If it's non-null
/// and we wouldn't otherwise try to reference a pass_object_size parameter,
/// we'll call `cir.objsize` on EmittedE, rather than emitting E.
mlir::Value CIRGenFunction::emitBuiltinObjectSize(const Expr *E, unsigned Type,
                                                  cir::IntType ResType,
                                                  mlir::Value EmittedE,
                                                  bool IsDynamic) {
  // We need to reference an argument if the pointer is a parameter with the
  // pass_object_size attribute.
  if (auto *D = dyn_cast<DeclRefExpr>(E->IgnoreParenImpCasts())) {
    auto *Param = dyn_cast<ParmVarDecl>(D->getDecl());
    auto *PS = D->getDecl()->getAttr<PassObjectSizeAttr>();
    if (Param != nullptr && PS != nullptr &&
        areBOSTypesCompatible(PS->getType(), Type)) {
      auto Iter = SizeArguments.find(Param);
      assert(Iter != SizeArguments.end());

      const ImplicitParamDecl *D = Iter->second;
      auto DIter = LocalDeclMap.find(D);
      assert(DIter != LocalDeclMap.end());

      return emitLoadOfScalar(DIter->second, /*Volatile=*/false,
                              getContext().getSizeType(), E->getBeginLoc());
    }
  }

  // LLVM can't handle Type=3 appropriately, and __builtin_object_size shouldn't
  // evaluate E for side-effects. In either case, just like original LLVM
  // lowering, we shouldn't lower to `cir.objsize`.
  if (Type == 3 || (!EmittedE && E->HasSideEffects(getContext())))
    llvm_unreachable("NYI");

  auto Ptr = EmittedE ? EmittedE : emitScalarExpr(E);
  assert(mlir::isa<cir::PointerType>(Ptr.getType()) &&
         "Non-pointer passed to __builtin_object_size?");

  // LLVM intrinsics (which CIR lowers to at some point, only supports 0
  // and 2, account for that right now.
  cir::SizeInfoType sizeInfoTy =
      ((Type & 2) != 0) ? cir::SizeInfoType::min : cir::SizeInfoType::max;
  // TODO(cir): Heads up for LLVM lowering, For GCC compatibility,
  // __builtin_object_size treat NULL as unknown size.
  return builder.create<cir::ObjSizeOp>(getLoc(E->getSourceRange()), ResType,
                                        Ptr, sizeInfoTy, IsDynamic);
}

mlir::Value CIRGenFunction::evaluateOrEmitBuiltinObjectSize(
    const Expr *E, unsigned Type, cir::IntType ResType, mlir::Value EmittedE,
    bool IsDynamic) {
  uint64_t ObjectSize;
  if (!E->tryEvaluateObjectSize(ObjectSize, getContext(), Type))
    return emitBuiltinObjectSize(E, Type, ResType, EmittedE, IsDynamic);
  return builder.getConstInt(getLoc(E->getSourceRange()), ResType, ObjectSize);
}

/// Given a builtin id for a function like "__builtin_fabsf", return a Function*
/// for "fabsf".
cir::FuncOp CIRGenModule::getBuiltinLibFunction(const FunctionDecl *FD,
                                                unsigned BuiltinID) {
  assert(astContext.BuiltinInfo.isLibFunction(BuiltinID));

  // Get the name, skip over the __builtin_ prefix (if necessary).
  StringRef Name;
  GlobalDecl D(FD);

  // TODO: This list should be expanded or refactored after all GCC-compatible
  // std libcall builtins are implemented.
  static SmallDenseMap<unsigned, StringRef, 64> F128Builtins{
      {Builtin::BI__builtin___fprintf_chk, "__fprintf_chkieee128"},
      {Builtin::BI__builtin___printf_chk, "__printf_chkieee128"},
      {Builtin::BI__builtin___snprintf_chk, "__snprintf_chkieee128"},
      {Builtin::BI__builtin___sprintf_chk, "__sprintf_chkieee128"},
      {Builtin::BI__builtin___vfprintf_chk, "__vfprintf_chkieee128"},
      {Builtin::BI__builtin___vprintf_chk, "__vprintf_chkieee128"},
      {Builtin::BI__builtin___vsnprintf_chk, "__vsnprintf_chkieee128"},
      {Builtin::BI__builtin___vsprintf_chk, "__vsprintf_chkieee128"},
      {Builtin::BI__builtin_fprintf, "__fprintfieee128"},
      {Builtin::BI__builtin_printf, "__printfieee128"},
      {Builtin::BI__builtin_snprintf, "__snprintfieee128"},
      {Builtin::BI__builtin_sprintf, "__sprintfieee128"},
      {Builtin::BI__builtin_vfprintf, "__vfprintfieee128"},
      {Builtin::BI__builtin_vprintf, "__vprintfieee128"},
      {Builtin::BI__builtin_vsnprintf, "__vsnprintfieee128"},
      {Builtin::BI__builtin_vsprintf, "__vsprintfieee128"},
      {Builtin::BI__builtin_fscanf, "__fscanfieee128"},
      {Builtin::BI__builtin_scanf, "__scanfieee128"},
      {Builtin::BI__builtin_sscanf, "__sscanfieee128"},
      {Builtin::BI__builtin_vfscanf, "__vfscanfieee128"},
      {Builtin::BI__builtin_vscanf, "__vscanfieee128"},
      {Builtin::BI__builtin_vsscanf, "__vsscanfieee128"},
      {Builtin::BI__builtin_nexttowardf128, "__nexttowardieee128"},
  };

  // The AIX library functions frexpl, ldexpl, and modfl are for 128-bit
  // IBM 'long double' (i.e. __ibm128). Map to the 'double' versions
  // if it is 64-bit 'long double' mode.
  static SmallDenseMap<unsigned, StringRef, 4> AIXLongDouble64Builtins{
      {Builtin::BI__builtin_frexpl, "frexp"},
      {Builtin::BI__builtin_ldexpl, "ldexp"},
      {Builtin::BI__builtin_modfl, "modf"},
  };

  // If the builtin has been declared explicitly with an assembler label,
  // use the mangled name. This differs from the plain label on platforms
  // that prefix labels.
  if (FD->hasAttr<AsmLabelAttr>())
    Name = getMangledName(D);
  else {
    // TODO: This mutation should also be applied to other targets other than
    // PPC, after backend supports IEEE 128-bit style libcalls.
    if (getTriple().isPPC64() &&
        &getTarget().getLongDoubleFormat() == &llvm::APFloat::IEEEquad() &&
        F128Builtins.find(BuiltinID) != F128Builtins.end())
      Name = F128Builtins[BuiltinID];
    else if (getTriple().isOSAIX() &&
             &getTarget().getLongDoubleFormat() ==
                 &llvm::APFloat::IEEEdouble() &&
             AIXLongDouble64Builtins.find(BuiltinID) !=
                 AIXLongDouble64Builtins.end())
      Name = AIXLongDouble64Builtins[BuiltinID];
    else
      Name = astContext.BuiltinInfo.getName(BuiltinID).substr(10);
  }

  auto Ty = convertType(FD->getType());
  return GetOrCreateCIRFunction(Name, Ty, D, /*ForVTable=*/false);
}

//===----------------------------------------------------------------------===//
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

#include "CIRGenCall.h"
#include "CIRGenFunction.h"
#include "CIRGenModule.h"
#include "CIRGenValue.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "clang/AST/DeclBase.h"
#include "clang/AST/Expr.h"
#include "clang/AST/GlobalDecl.h"
#include "clang/Basic/Builtins.h"
#include "clang/Basic/DiagnosticFrontend.h"
#include "clang/Basic/OperatorKinds.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "clang/CIR/MissingFeatures.h"
#include "llvm/Support/ErrorHandling.h"

using namespace clang;
using namespace clang::CIRGen;
using namespace llvm;

static RValue emitLibraryCall(CIRGenFunction &cgf, const FunctionDecl *fd,
                              const CallExpr *e, mlir::Operation *calleeValue) {
  CIRGenCallee callee = CIRGenCallee::forDirect(calleeValue, GlobalDecl(fd));
  return cgf.emitCall(e->getCallee()->getType(), callee, e, ReturnValueSlot());
}

template <typename Op>
static RValue emitBuiltinBitOp(CIRGenFunction &cgf, const CallExpr *e,
                               bool poisonZero = false) {
  assert(!cir::MissingFeatures::builtinCheckKind());

  mlir::Value arg = cgf.emitScalarExpr(e->getArg(0));
  CIRGenBuilderTy &builder = cgf.getBuilder();

  Op op;
  if constexpr (std::is_same_v<Op, cir::BitClzOp> ||
                std::is_same_v<Op, cir::BitCtzOp>)
    op = Op::create(builder, cgf.getLoc(e->getSourceRange()), arg, poisonZero);
  else
    op = Op::create(builder, cgf.getLoc(e->getSourceRange()), arg);

  mlir::Value result = op.getResult();
  mlir::Type exprTy = cgf.convertType(e->getType());
  if (exprTy != result.getType())
    result = builder.createIntCast(result, exprTy);

  return RValue::get(result);
}

/// Emit the conversions required to turn the given value into an
/// integer of the given size.
static mlir::Value emitToInt(CIRGenFunction &cgf, mlir::Value v, QualType t,
                             cir::IntType intType) {
  v = cgf.emitToMemory(v, t);

  if (mlir::isa<cir::PointerType>(v.getType()))
    return cgf.getBuilder().createPtrToInt(v, intType);

  assert(v.getType() == intType);
  return v;
}

static mlir::Value emitFromInt(CIRGenFunction &cgf, mlir::Value v, QualType t,
                               mlir::Type resultType) {
  v = cgf.emitFromMemory(v, t);

  if (mlir::isa<cir::PointerType>(resultType))
    return cgf.getBuilder().createIntToPtr(v, resultType);

  assert(v.getType() == resultType);
  return v;
}

static Address checkAtomicAlignment(CIRGenFunction &cgf, const CallExpr *e) {
  ASTContext &astContext = cgf.getContext();
  Address ptr = cgf.emitPointerWithAlignment(e->getArg(0));
  unsigned bytes =
      mlir::isa<cir::PointerType>(ptr.getElementType())
          ? astContext.getTypeSizeInChars(astContext.VoidPtrTy).getQuantity()
          : cgf.cgm.getDataLayout().getTypeSizeInBits(ptr.getElementType()) /
                cgf.cgm.getASTContext().getCharWidth();

  unsigned align = ptr.getAlignment().getQuantity();
  if (align % bytes != 0) {
    DiagnosticsEngine &diags = cgf.cgm.getDiags();
    diags.Report(e->getBeginLoc(), diag::warn_sync_op_misaligned);
    // Force address to be at least naturally-aligned.
    return ptr.withAlignment(CharUnits::fromQuantity(bytes));
  }
  return ptr;
}

/// Utility to insert an atomic instruction based on Intrinsic::ID
/// and the expression node.
static mlir::Value makeBinaryAtomicValue(
    CIRGenFunction &cgf, cir::AtomicFetchKind kind, const CallExpr *expr,
    mlir::Type *originalArgType, mlir::Value *emittedArgValue = nullptr,
    cir::MemOrder ordering = cir::MemOrder::SequentiallyConsistent) {

  QualType type = expr->getType();
  QualType ptrType = expr->getArg(0)->getType();

  assert(ptrType->isPointerType());
  assert(
      cgf.getContext().hasSameUnqualifiedType(type, ptrType->getPointeeType()));
  assert(cgf.getContext().hasSameUnqualifiedType(type,
                                                 expr->getArg(1)->getType()));

  Address destAddr = checkAtomicAlignment(cgf, expr);
  CIRGenBuilderTy &builder = cgf.getBuilder();

  mlir::Value val = cgf.emitScalarExpr(expr->getArg(1));
  mlir::Type valueType = val.getType();
  mlir::Value destValue = destAddr.emitRawPointer();

  if (ptrType->getPointeeType()->isPointerType()) {
    // Pointer to pointer
    // `cir.atomic.fetch` expects a pointer to an integer type, so we cast
    // ptr<ptr<T>> to ptr<intPtrSize>
    cir::IntType ptrSizeInt =
        builder.getSIntNTy(cgf.getContext().getTypeSize(ptrType));
    destValue =
        builder.createBitcast(destValue, builder.getPointerTo(ptrSizeInt));
    val = emitToInt(cgf, val, type, ptrSizeInt);
  } else {
    // Pointer to integer type
    cir::IntType intType =
        ptrType->getPointeeType()->isUnsignedIntegerType()
            ? builder.getUIntNTy(cgf.getContext().getTypeSize(type))
            : builder.getSIntNTy(cgf.getContext().getTypeSize(type));
    val = emitToInt(cgf, val, type, intType);
  }

  // This output argument is needed for post atomic fetch operations
  // that calculate the result of the operation as return value of
  // <binop>_and_fetch builtins. The `AtomicFetch` operation only updates the
  // memory location and returns the old value.
  if (emittedArgValue) {
    *emittedArgValue = val;
    *originalArgType = valueType;
  }

  auto rmwi = cir::AtomicFetchOp::create(
      builder, cgf.getLoc(expr->getSourceRange()), destValue, val, kind,
      ordering, false, /* is volatile */
      true);           /* fetch first */
  return rmwi->getResult(0);
}

static RValue emitBinaryAtomicPost(CIRGenFunction &cgf,
                                   cir::AtomicFetchKind atomicOpkind,
                                   const CallExpr *e, cir::BinOpKind binopKind,
                                   bool invert = false) {
  mlir::Value emittedArgValue;
  mlir::Type originalArgType;
  clang::QualType typ = e->getType();
  mlir::Value result = makeBinaryAtomicValue(
      cgf, atomicOpkind, e, &originalArgType, &emittedArgValue);
  clang::CIRGen::CIRGenBuilderTy &builder = cgf.getBuilder();
  result = cir::BinOp::create(builder, result.getLoc(), binopKind, result,
                              emittedArgValue);

  if (invert)
    result = cir::UnaryOp::create(builder, result.getLoc(),
                                  cir::UnaryOpKind::Not, result);

  result = emitFromInt(cgf, result, typ, originalArgType);
  return RValue::get(result);
}

static void emitAtomicFenceOp(CIRGenFunction &cgf, const CallExpr *expr,
                              cir::SyncScopeKind syncScope) {
  CIRGenBuilderTy &builder = cgf.getBuilder();
  mlir::Location loc = cgf.getLoc(expr->getSourceRange());

  auto emitAtomicOpCallBackFn = [&](cir::MemOrder memOrder) {
    cir::AtomicFenceOp::create(
        builder, loc, memOrder,
        cir::SyncScopeKindAttr::get(&cgf.getMLIRContext(), syncScope));
  };

  cgf.emitAtomicExprWithMemOrder(expr->getArg(0), /*isStore*/ false,
                                 /*isLoad*/ false, /*isFence*/ true,
                                 emitAtomicOpCallBackFn);
}

namespace {
struct WidthAndSignedness {
  unsigned width;
  bool isSigned;
};
} // namespace

static WidthAndSignedness
getIntegerWidthAndSignedness(const clang::ASTContext &astContext,
                             const clang::QualType type) {
  assert(type->isIntegerType() && "Given type is not an integer.");
  unsigned width = type->isBooleanType()  ? 1
                   : type->isBitIntType() ? astContext.getIntWidth(type)
                                          : astContext.getTypeInfo(type).Width;
  bool isSigned = type->isSignedIntegerType();
  return {width, isSigned};
}

// Given one or more integer types, this function produces an integer type that
// encompasses them: any value in one of the given types could be expressed in
// the encompassing type.
static struct WidthAndSignedness
EncompassingIntegerType(ArrayRef<struct WidthAndSignedness> types) {
  assert(types.size() > 0 && "Empty list of types.");

  // If any of the given types is signed, we must return a signed type.
  bool isSigned = llvm::any_of(types, [](const auto &t) { return t.isSigned; });

  // The encompassing type must have a width greater than or equal to the width
  // of the specified types.  Additionally, if the encompassing type is signed,
  // its width must be strictly greater than the width of any unsigned types
  // given.
  unsigned width = 0;
  for (const auto &type : types)
    width = std::max(width, type.width + (isSigned && !type.isSigned));

  return {width, isSigned};
}

RValue CIRGenFunction::emitRotate(const CallExpr *e, bool isRotateLeft) {
  mlir::Value input = emitScalarExpr(e->getArg(0));
  mlir::Value amount = emitScalarExpr(e->getArg(1));

  // TODO(cir): MSVC flavor bit rotate builtins use different types for input
  // and amount, but cir.rotate requires them to have the same type. Cast amount
  // to the type of input when necessary.
  assert(!cir::MissingFeatures::msvcBuiltins());

  auto r = cir::RotateOp::create(builder, getLoc(e->getSourceRange()), input,
                                 amount, isRotateLeft);
  return RValue::get(r);
}

template <class Operation>
static RValue emitUnaryMaybeConstrainedFPBuiltin(CIRGenFunction &cgf,
                                                 const CallExpr &e) {
  mlir::Value arg = cgf.emitScalarExpr(e.getArg(0));

  assert(!cir::MissingFeatures::cgFPOptionsRAII());
  assert(!cir::MissingFeatures::fpConstraints());

  auto call =
      Operation::create(cgf.getBuilder(), arg.getLoc(), arg.getType(), arg);
  return RValue::get(call->getResult(0));
}

template <class Operation>
static RValue emitUnaryFPBuiltin(CIRGenFunction &cgf, const CallExpr &e) {
  mlir::Value arg = cgf.emitScalarExpr(e.getArg(0));
  auto call =
      Operation::create(cgf.getBuilder(), arg.getLoc(), arg.getType(), arg);
  return RValue::get(call->getResult(0));
}

static RValue errorBuiltinNYI(CIRGenFunction &cgf, const CallExpr *e,
                              unsigned builtinID) {

  if (cgf.getContext().BuiltinInfo.isLibFunction(builtinID)) {
    cgf.cgm.errorNYI(
        e->getSourceRange(),
        std::string("unimplemented X86 library function builtin call: ") +
            cgf.getContext().BuiltinInfo.getName(builtinID));
  } else {
    cgf.cgm.errorNYI(e->getSourceRange(),
                     std::string("unimplemented X86 builtin call: ") +
                         cgf.getContext().BuiltinInfo.getName(builtinID));
  }

  return cgf.getUndefRValue(e->getType());
}

static RValue emitBuiltinAlloca(CIRGenFunction &cgf, const CallExpr *e,
                                unsigned builtinID) {
  assert(builtinID == Builtin::BI__builtin_alloca ||
         builtinID == Builtin::BI__builtin_alloca_uninitialized ||
         builtinID == Builtin::BIalloca || builtinID == Builtin::BI_alloca);

  // Get alloca size input
  mlir::Value size = cgf.emitScalarExpr(e->getArg(0));

  // The alignment of the alloca should correspond to __BIGGEST_ALIGNMENT__.
  const TargetInfo &ti = cgf.getContext().getTargetInfo();
  const CharUnits suitableAlignmentInBytes =
      cgf.getContext().toCharUnitsFromBits(ti.getSuitableAlign());

  // Emit the alloca op with type `u8 *` to match the semantics of
  // `llvm.alloca`. We later bitcast the type to `void *` to match the
  // semantics of C/C++
  // FIXME(cir): It may make sense to allow AllocaOp of type `u8` to return a
  // pointer of type `void *`. This will require a change to the allocaOp
  // verifier.
  CIRGenBuilderTy &builder = cgf.getBuilder();
  mlir::Value allocaAddr = builder.createAlloca(
      cgf.getLoc(e->getSourceRange()), builder.getUInt8PtrTy(),
      builder.getUInt8Ty(), "bi_alloca", suitableAlignmentInBytes, size);

  // Initialize the allocated buffer if required.
  if (builtinID != Builtin::BI__builtin_alloca_uninitialized) {
    // Initialize the alloca with the given size and alignment according to
    // the lang opts. Only the trivial non-initialization is supported for
    // now.

    switch (cgf.getLangOpts().getTrivialAutoVarInit()) {
    case LangOptions::TrivialAutoVarInitKind::Uninitialized:
      // Nothing to initialize.
      break;
    case LangOptions::TrivialAutoVarInitKind::Zero:
    case LangOptions::TrivialAutoVarInitKind::Pattern:
      cgf.cgm.errorNYI("trivial auto var init");
      break;
    }
  }

  // An alloca will always return a pointer to the alloca (stack) address
  // space. This address space need not be the same as the AST / Language
  // default (e.g. in C / C++ auto vars are in the generic address space). At
  // the AST level this is handled within CreateTempAlloca et al., but for the
  // builtin / dynamic alloca we have to handle it here.

  if (!cir::isMatchingAddressSpace(
          cgf.getCIRAllocaAddressSpace(),
          e->getType()->getPointeeType().getAddressSpace())) {
    cgf.cgm.errorNYI(e->getSourceRange(),
                     "Non-default address space for alloca");
  }

  // Bitcast the alloca to the expected type.
  return RValue::get(builder.createBitcast(
      allocaAddr, builder.getVoidPtrTy(cgf.getCIRAllocaAddressSpace())));
}

static bool shouldCIREmitFPMathIntrinsic(CIRGenFunction &cgf, const CallExpr *e,
                                         unsigned builtinID) {
  std::optional<bool> errnoOverriden;
  // ErrnoOverriden is true if math-errno is overriden via the
  // '#pragma float_control(precise, on)'. This pragma disables fast-math,
  // which implies math-errno.
  if (e->hasStoredFPFeatures()) {
    FPOptionsOverride op = e->getFPFeatures();
    if (op.hasMathErrnoOverride())
      errnoOverriden = op.getMathErrnoOverride();
  }
  // True if 'attribute__((optnone))' is used. This attribute overrides
  // fast-math which implies math-errno.
  bool optNone =
      cgf.curFuncDecl && cgf.curFuncDecl->hasAttr<OptimizeNoneAttr>();
  bool isOptimizationEnabled = cgf.cgm.getCodeGenOpts().OptimizationLevel != 0;
  bool generateFPMathIntrinsics =
      cgf.getContext().BuiltinInfo.shouldGenerateFPMathIntrinsic(
          builtinID, cgf.cgm.getTriple(), errnoOverriden,
          cgf.getLangOpts().MathErrno, optNone, isOptimizationEnabled);
  return generateFPMathIntrinsics;
}

static RValue tryEmitFPMathIntrinsic(CIRGenFunction &cgf, const CallExpr *e,
                                     unsigned builtinID) {
  assert(!cir::MissingFeatures::fastMathFlags());
  switch (builtinID) {
  case Builtin::BIacos:
  case Builtin::BIacosf:
  case Builtin::BIacosl:
  case Builtin::BI__builtin_acos:
  case Builtin::BI__builtin_acosf:
  case Builtin::BI__builtin_acosf16:
  case Builtin::BI__builtin_acosl:
  case Builtin::BI__builtin_acosf128:
  case Builtin::BI__builtin_elementwise_acos:
  case Builtin::BIasin:
  case Builtin::BIasinf:
  case Builtin::BIasinl:
  case Builtin::BI__builtin_asin:
  case Builtin::BI__builtin_asinf:
  case Builtin::BI__builtin_asinf16:
  case Builtin::BI__builtin_asinl:
  case Builtin::BI__builtin_asinf128:
  case Builtin::BI__builtin_elementwise_asin:
  case Builtin::BIatan:
  case Builtin::BIatanf:
  case Builtin::BIatanl:
  case Builtin::BI__builtin_atan:
  case Builtin::BI__builtin_atanf:
  case Builtin::BI__builtin_atanf16:
  case Builtin::BI__builtin_atanl:
  case Builtin::BI__builtin_atanf128:
  case Builtin::BI__builtin_elementwise_atan:
  case Builtin::BIatan2:
  case Builtin::BIatan2f:
  case Builtin::BIatan2l:
  case Builtin::BI__builtin_atan2:
  case Builtin::BI__builtin_atan2f:
  case Builtin::BI__builtin_atan2f16:
  case Builtin::BI__builtin_atan2l:
  case Builtin::BI__builtin_atan2f128:
  case Builtin::BI__builtin_elementwise_atan2:
    return RValue::getIgnored();
  case Builtin::BIceil:
  case Builtin::BIceilf:
  case Builtin::BIceill:
  case Builtin::BI__builtin_ceil:
  case Builtin::BI__builtin_ceilf:
  case Builtin::BI__builtin_ceilf16:
  case Builtin::BI__builtin_ceill:
  case Builtin::BI__builtin_ceilf128:
    return emitUnaryMaybeConstrainedFPBuiltin<cir::CeilOp>(cgf, *e);
  case Builtin::BI__builtin_elementwise_ceil:
  case Builtin::BIcopysign:
  case Builtin::BIcopysignf:
  case Builtin::BIcopysignl:
  case Builtin::BI__builtin_copysign:
  case Builtin::BI__builtin_copysignf:
  case Builtin::BI__builtin_copysignf16:
  case Builtin::BI__builtin_copysignl:
  case Builtin::BI__builtin_copysignf128:
    return RValue::getIgnored();
  case Builtin::BIcos:
  case Builtin::BIcosf:
  case Builtin::BIcosl:
  case Builtin::BI__builtin_cos:
  case Builtin::BI__builtin_cosf:
  case Builtin::BI__builtin_cosf16:
  case Builtin::BI__builtin_cosl:
  case Builtin::BI__builtin_cosf128:
    return emitUnaryMaybeConstrainedFPBuiltin<cir::CosOp>(cgf, *e);
  case Builtin::BI__builtin_elementwise_cos:
  case Builtin::BIcosh:
  case Builtin::BIcoshf:
  case Builtin::BIcoshl:
  case Builtin::BI__builtin_cosh:
  case Builtin::BI__builtin_coshf:
  case Builtin::BI__builtin_coshf16:
  case Builtin::BI__builtin_coshl:
  case Builtin::BI__builtin_coshf128:
  case Builtin::BI__builtin_elementwise_cosh:
    return RValue::getIgnored();
  case Builtin::BIexp:
  case Builtin::BIexpf:
  case Builtin::BIexpl:
  case Builtin::BI__builtin_exp:
  case Builtin::BI__builtin_expf:
  case Builtin::BI__builtin_expf16:
  case Builtin::BI__builtin_expl:
  case Builtin::BI__builtin_expf128:
    return emitUnaryMaybeConstrainedFPBuiltin<cir::ExpOp>(cgf, *e);
  case Builtin::BI__builtin_elementwise_exp:
    return RValue::getIgnored();
  case Builtin::BIexp2:
  case Builtin::BIexp2f:
  case Builtin::BIexp2l:
  case Builtin::BI__builtin_exp2:
  case Builtin::BI__builtin_exp2f:
  case Builtin::BI__builtin_exp2f16:
  case Builtin::BI__builtin_exp2l:
  case Builtin::BI__builtin_exp2f128:
    return emitUnaryMaybeConstrainedFPBuiltin<cir::Exp2Op>(cgf, *e);
  case Builtin::BI__builtin_elementwise_exp2:
  case Builtin::BI__builtin_exp10:
  case Builtin::BI__builtin_exp10f:
  case Builtin::BI__builtin_exp10f16:
  case Builtin::BI__builtin_exp10l:
  case Builtin::BI__builtin_exp10f128:
  case Builtin::BI__builtin_elementwise_exp10:
    return RValue::getIgnored();
  case Builtin::BIfabs:
  case Builtin::BIfabsf:
  case Builtin::BIfabsl:
  case Builtin::BI__builtin_fabs:
  case Builtin::BI__builtin_fabsf:
  case Builtin::BI__builtin_fabsf16:
  case Builtin::BI__builtin_fabsl:
  case Builtin::BI__builtin_fabsf128:
    return emitUnaryMaybeConstrainedFPBuiltin<cir::FAbsOp>(cgf, *e);
  case Builtin::BIfloor:
  case Builtin::BIfloorf:
  case Builtin::BIfloorl:
  case Builtin::BI__builtin_floor:
  case Builtin::BI__builtin_floorf:
  case Builtin::BI__builtin_floorf16:
  case Builtin::BI__builtin_floorl:
  case Builtin::BI__builtin_floorf128:
    return emitUnaryMaybeConstrainedFPBuiltin<cir::FloorOp>(cgf, *e);
  case Builtin::BI__builtin_elementwise_floor:
  case Builtin::BIfma:
  case Builtin::BIfmaf:
  case Builtin::BIfmal:
  case Builtin::BI__builtin_fma:
  case Builtin::BI__builtin_fmaf:
  case Builtin::BI__builtin_fmaf16:
  case Builtin::BI__builtin_fmal:
  case Builtin::BI__builtin_fmaf128:
  case Builtin::BI__builtin_elementwise_fma:
  case Builtin::BIfmax:
  case Builtin::BIfmaxf:
  case Builtin::BIfmaxl:
  case Builtin::BI__builtin_fmax:
  case Builtin::BI__builtin_fmaxf:
  case Builtin::BI__builtin_fmaxf16:
  case Builtin::BI__builtin_fmaxl:
  case Builtin::BI__builtin_fmaxf128:
  case Builtin::BIfmin:
  case Builtin::BIfminf:
  case Builtin::BIfminl:
  case Builtin::BI__builtin_fmin:
  case Builtin::BI__builtin_fminf:
  case Builtin::BI__builtin_fminf16:
  case Builtin::BI__builtin_fminl:
  case Builtin::BI__builtin_fminf128:
  case Builtin::BIfmaximum_num:
  case Builtin::BIfmaximum_numf:
  case Builtin::BIfmaximum_numl:
  case Builtin::BI__builtin_fmaximum_num:
  case Builtin::BI__builtin_fmaximum_numf:
  case Builtin::BI__builtin_fmaximum_numf16:
  case Builtin::BI__builtin_fmaximum_numl:
  case Builtin::BI__builtin_fmaximum_numf128:
  case Builtin::BIfminimum_num:
  case Builtin::BIfminimum_numf:
  case Builtin::BIfminimum_numl:
  case Builtin::BI__builtin_fminimum_num:
  case Builtin::BI__builtin_fminimum_numf:
  case Builtin::BI__builtin_fminimum_numf16:
  case Builtin::BI__builtin_fminimum_numl:
  case Builtin::BI__builtin_fminimum_numf128:
  case Builtin::BIfmod:
  case Builtin::BIfmodf:
  case Builtin::BIfmodl:
  case Builtin::BI__builtin_fmod:
  case Builtin::BI__builtin_fmodf:
  case Builtin::BI__builtin_fmodf16:
  case Builtin::BI__builtin_fmodl:
  case Builtin::BI__builtin_fmodf128:
  case Builtin::BI__builtin_elementwise_fmod:
  case Builtin::BIlog:
  case Builtin::BIlogf:
  case Builtin::BIlogl:
  case Builtin::BI__builtin_log:
  case Builtin::BI__builtin_logf:
  case Builtin::BI__builtin_logf16:
  case Builtin::BI__builtin_logl:
  case Builtin::BI__builtin_logf128:
  case Builtin::BI__builtin_elementwise_log:
  case Builtin::BIlog10:
  case Builtin::BIlog10f:
  case Builtin::BIlog10l:
  case Builtin::BI__builtin_log10:
  case Builtin::BI__builtin_log10f:
  case Builtin::BI__builtin_log10f16:
  case Builtin::BI__builtin_log10l:
  case Builtin::BI__builtin_log10f128:
  case Builtin::BI__builtin_elementwise_log10:
  case Builtin::BIlog2:
  case Builtin::BIlog2f:
  case Builtin::BIlog2l:
  case Builtin::BI__builtin_log2:
  case Builtin::BI__builtin_log2f:
  case Builtin::BI__builtin_log2f16:
  case Builtin::BI__builtin_log2l:
  case Builtin::BI__builtin_log2f128:
  case Builtin::BI__builtin_elementwise_log2:
  case Builtin::BInearbyint:
  case Builtin::BInearbyintf:
  case Builtin::BInearbyintl:
  case Builtin::BI__builtin_nearbyint:
  case Builtin::BI__builtin_nearbyintf:
  case Builtin::BI__builtin_nearbyintl:
  case Builtin::BI__builtin_nearbyintf128:
  case Builtin::BI__builtin_elementwise_nearbyint:
  case Builtin::BIpow:
  case Builtin::BIpowf:
  case Builtin::BIpowl:
  case Builtin::BI__builtin_pow:
  case Builtin::BI__builtin_powf:
  case Builtin::BI__builtin_powf16:
  case Builtin::BI__builtin_powl:
  case Builtin::BI__builtin_powf128:
  case Builtin::BI__builtin_elementwise_pow:
  case Builtin::BIrint:
  case Builtin::BIrintf:
  case Builtin::BIrintl:
  case Builtin::BI__builtin_rint:
  case Builtin::BI__builtin_rintf:
  case Builtin::BI__builtin_rintf16:
  case Builtin::BI__builtin_rintl:
  case Builtin::BI__builtin_rintf128:
  case Builtin::BI__builtin_elementwise_rint:
  case Builtin::BIround:
  case Builtin::BIroundf:
  case Builtin::BIroundl:
  case Builtin::BI__builtin_round:
  case Builtin::BI__builtin_roundf:
  case Builtin::BI__builtin_roundf16:
  case Builtin::BI__builtin_roundl:
  case Builtin::BI__builtin_roundf128:
  case Builtin::BI__builtin_elementwise_round:
  case Builtin::BIroundeven:
  case Builtin::BIroundevenf:
  case Builtin::BIroundevenl:
  case Builtin::BI__builtin_roundeven:
  case Builtin::BI__builtin_roundevenf:
  case Builtin::BI__builtin_roundevenf16:
  case Builtin::BI__builtin_roundevenl:
  case Builtin::BI__builtin_roundevenf128:
  case Builtin::BI__builtin_elementwise_roundeven:
  case Builtin::BIsin:
  case Builtin::BIsinf:
  case Builtin::BIsinl:
  case Builtin::BI__builtin_sin:
  case Builtin::BI__builtin_sinf:
  case Builtin::BI__builtin_sinf16:
  case Builtin::BI__builtin_sinl:
  case Builtin::BI__builtin_sinf128:
  case Builtin::BI__builtin_elementwise_sin:
  case Builtin::BIsinh:
  case Builtin::BIsinhf:
  case Builtin::BIsinhl:
  case Builtin::BI__builtin_sinh:
  case Builtin::BI__builtin_sinhf:
  case Builtin::BI__builtin_sinhf16:
  case Builtin::BI__builtin_sinhl:
  case Builtin::BI__builtin_sinhf128:
  case Builtin::BI__builtin_elementwise_sinh:
  case Builtin::BI__builtin_sincospi:
  case Builtin::BI__builtin_sincospif:
  case Builtin::BI__builtin_sincospil:
  case Builtin::BIsincos:
  case Builtin::BIsincosf:
  case Builtin::BIsincosl:
  case Builtin::BI__builtin_sincos:
  case Builtin::BI__builtin_sincosf:
  case Builtin::BI__builtin_sincosf16:
  case Builtin::BI__builtin_sincosl:
  case Builtin::BI__builtin_sincosf128:
  case Builtin::BIsqrt:
  case Builtin::BIsqrtf:
  case Builtin::BIsqrtl:
  case Builtin::BI__builtin_sqrt:
  case Builtin::BI__builtin_sqrtf:
  case Builtin::BI__builtin_sqrtf16:
  case Builtin::BI__builtin_sqrtl:
  case Builtin::BI__builtin_sqrtf128:
  case Builtin::BI__builtin_elementwise_sqrt:
  case Builtin::BItan:
  case Builtin::BItanf:
  case Builtin::BItanl:
  case Builtin::BI__builtin_tan:
  case Builtin::BI__builtin_tanf:
  case Builtin::BI__builtin_tanf16:
  case Builtin::BI__builtin_tanl:
  case Builtin::BI__builtin_tanf128:
  case Builtin::BI__builtin_elementwise_tan:
  case Builtin::BItanh:
  case Builtin::BItanhf:
  case Builtin::BItanhl:
  case Builtin::BI__builtin_tanh:
  case Builtin::BI__builtin_tanhf:
  case Builtin::BI__builtin_tanhf16:
  case Builtin::BI__builtin_tanhl:
  case Builtin::BI__builtin_tanhf128:
  case Builtin::BI__builtin_elementwise_tanh:
  case Builtin::BItrunc:
  case Builtin::BItruncf:
  case Builtin::BItruncl:
  case Builtin::BI__builtin_trunc:
  case Builtin::BI__builtin_truncf:
  case Builtin::BI__builtin_truncf16:
  case Builtin::BI__builtin_truncl:
  case Builtin::BI__builtin_truncf128:
  case Builtin::BI__builtin_elementwise_trunc:
  case Builtin::BIlround:
  case Builtin::BIlroundf:
  case Builtin::BIlroundl:
  case Builtin::BI__builtin_lround:
  case Builtin::BI__builtin_lroundf:
  case Builtin::BI__builtin_lroundl:
  case Builtin::BI__builtin_lroundf128:
  case Builtin::BIllround:
  case Builtin::BIllroundf:
  case Builtin::BIllroundl:
  case Builtin::BI__builtin_llround:
  case Builtin::BI__builtin_llroundf:
  case Builtin::BI__builtin_llroundl:
  case Builtin::BI__builtin_llroundf128:
  case Builtin::BIlrint:
  case Builtin::BIlrintf:
  case Builtin::BIlrintl:
  case Builtin::BI__builtin_lrint:
  case Builtin::BI__builtin_lrintf:
  case Builtin::BI__builtin_lrintl:
  case Builtin::BI__builtin_lrintf128:
  case Builtin::BIllrint:
  case Builtin::BIllrintf:
  case Builtin::BIllrintl:
  case Builtin::BI__builtin_llrint:
  case Builtin::BI__builtin_llrintf:
  case Builtin::BI__builtin_llrintl:
  case Builtin::BI__builtin_llrintf128:
  case Builtin::BI__builtin_ldexp:
  case Builtin::BI__builtin_ldexpf:
  case Builtin::BI__builtin_ldexpl:
  case Builtin::BI__builtin_ldexpf16:
  case Builtin::BI__builtin_ldexpf128:
  case Builtin::BI__builtin_elementwise_ldexp:
  default:
    break;
  }

  return RValue::getIgnored();
}

RValue CIRGenFunction::emitBuiltinExpr(const GlobalDecl &gd, unsigned builtinID,
                                       const CallExpr *e,
                                       ReturnValueSlot returnValue) {
  mlir::Location loc = getLoc(e->getSourceRange());

  // See if we can constant fold this builtin.  If so, don't emit it at all.
  // TODO: Extend this handling to all builtin calls that we can constant-fold.
  Expr::EvalResult result;
  if (e->isPRValue() && e->EvaluateAsRValue(result, cgm.getASTContext()) &&
      !result.hasSideEffects()) {
    if (result.Val.isInt())
      return RValue::get(builder.getConstInt(loc, result.Val.getInt()));
    if (result.Val.isFloat()) {
      // Note: we are using result type of CallExpr to determine the type of
      // the constant. Classic codegen uses the result value to determine the
      // type. We feel it should be Ok to use expression type because it is
      // hard to imagine a builtin function evaluates to a value that
      // over/underflows its own defined type.
      mlir::Type type = convertType(e->getType());
      return RValue::get(builder.getConstFP(loc, type, result.Val.getFloat()));
    }
  }

  const FunctionDecl *fd = gd.getDecl()->getAsFunction();

  assert(!cir::MissingFeatures::builtinCallF128());

  // If the builtin has been declared explicitly with an assembler label,
  // disable the specialized emitting below. Ideally we should communicate the
  // rename in IR, or at least avoid generating the intrinsic calls that are
  // likely to get lowered to the renamed library functions.
  unsigned builtinIDIfNoAsmLabel = fd->hasAttr<AsmLabelAttr>() ? 0 : builtinID;

  bool generateFPMathIntrinsics =
      shouldCIREmitFPMathIntrinsic(*this, e, builtinID);

  if (generateFPMathIntrinsics) {
    // Try to match the builtinID with a floating point math builtin.
    RValue rv = tryEmitFPMathIntrinsic(*this, e, builtinIDIfNoAsmLabel);

    // Return the result directly if a math intrinsic was generated.
    if (!rv.isIgnored()) {
      return rv;
    }
  }

  assert(!cir::MissingFeatures::builtinCall());

  switch (builtinIDIfNoAsmLabel) {
  default:
    break;

  // C stdarg builtins.
  case Builtin::BI__builtin_stdarg_start:
  case Builtin::BI__builtin_va_start:
  case Builtin::BI__va_start: {
    mlir::Value vaList = builtinID == Builtin::BI__va_start
                             ? emitScalarExpr(e->getArg(0))
                             : emitVAListRef(e->getArg(0)).getPointer();
    mlir::Value count = emitScalarExpr(e->getArg(1));
    emitVAStart(vaList, count);
    return {};
  }

  case Builtin::BI__builtin_va_end:
    emitVAEnd(emitVAListRef(e->getArg(0)).getPointer());
    return {};
  case Builtin::BI__builtin_va_copy: {
    mlir::Value dstPtr = emitVAListRef(e->getArg(0)).getPointer();
    mlir::Value srcPtr = emitVAListRef(e->getArg(1)).getPointer();
    cir::VACopyOp::create(builder, dstPtr.getLoc(), dstPtr, srcPtr);
    return {};
  }
  case Builtin::BI__assume:
  case Builtin::BI__builtin_assume: {
    if (e->getArg(0)->HasSideEffects(getContext()))
      return RValue::get(nullptr);

    mlir::Value argValue = emitCheckedArgForAssume(e->getArg(0));
    cir::AssumeOp::create(builder, loc, argValue);
    return RValue::get(nullptr);
  }

  case Builtin::BI__builtin_assume_separate_storage: {
    mlir::Value value0 = emitScalarExpr(e->getArg(0));
    mlir::Value value1 = emitScalarExpr(e->getArg(1));
    cir::AssumeSepStorageOp::create(builder, loc, value0, value1);
    return RValue::get(nullptr);
  }

  case Builtin::BI__builtin_assume_aligned: {
    const Expr *ptrExpr = e->getArg(0);
    mlir::Value ptrValue = emitScalarExpr(ptrExpr);
    mlir::Value offsetValue =
        (e->getNumArgs() > 2) ? emitScalarExpr(e->getArg(2)) : nullptr;

    std::optional<llvm::APSInt> alignment =
        e->getArg(1)->getIntegerConstantExpr(getContext());
    assert(alignment.has_value() &&
           "the second argument to __builtin_assume_aligned must be an "
           "integral constant expression");

    mlir::Value result =
        emitAlignmentAssumption(ptrValue, ptrExpr, ptrExpr->getExprLoc(),
                                alignment->getSExtValue(), offsetValue);
    return RValue::get(result);
  }

  case Builtin::BI__builtin_complex: {
    mlir::Value real = emitScalarExpr(e->getArg(0));
    mlir::Value imag = emitScalarExpr(e->getArg(1));
    mlir::Value complex = builder.createComplexCreate(loc, real, imag);
    return RValue::getComplex(complex);
  }

  case Builtin::BI__builtin_creal:
  case Builtin::BI__builtin_crealf:
  case Builtin::BI__builtin_creall:
  case Builtin::BIcreal:
  case Builtin::BIcrealf:
  case Builtin::BIcreall: {
    mlir::Value complex = emitComplexExpr(e->getArg(0));
    mlir::Value real = builder.createComplexReal(loc, complex);
    return RValue::get(real);
  }

  case Builtin::BI__builtin_cimag:
  case Builtin::BI__builtin_cimagf:
  case Builtin::BI__builtin_cimagl:
  case Builtin::BIcimag:
  case Builtin::BIcimagf:
  case Builtin::BIcimagl: {
    mlir::Value complex = emitComplexExpr(e->getArg(0));
    mlir::Value imag = builder.createComplexImag(loc, complex);
    return RValue::get(imag);
  }

  case Builtin::BI__builtin_conj:
  case Builtin::BI__builtin_conjf:
  case Builtin::BI__builtin_conjl:
  case Builtin::BIconj:
  case Builtin::BIconjf:
  case Builtin::BIconjl: {
    mlir::Value complex = emitComplexExpr(e->getArg(0));
    mlir::Value conj = builder.createUnaryOp(getLoc(e->getExprLoc()),
                                             cir::UnaryOpKind::Not, complex);
    return RValue::getComplex(conj);
  }

  case Builtin::BI__builtin_clrsb:
  case Builtin::BI__builtin_clrsbl:
  case Builtin::BI__builtin_clrsbll:
    return emitBuiltinBitOp<cir::BitClrsbOp>(*this, e);

  case Builtin::BI__builtin_ctzs:
  case Builtin::BI__builtin_ctz:
  case Builtin::BI__builtin_ctzl:
  case Builtin::BI__builtin_ctzll:
  case Builtin::BI__builtin_ctzg:
    assert(!cir::MissingFeatures::builtinCheckKind());
    return emitBuiltinBitOp<cir::BitCtzOp>(*this, e, /*poisonZero=*/true);

  case Builtin::BI__builtin_clzs:
  case Builtin::BI__builtin_clz:
  case Builtin::BI__builtin_clzl:
  case Builtin::BI__builtin_clzll:
  case Builtin::BI__builtin_clzg:
    assert(!cir::MissingFeatures::builtinCheckKind());
    return emitBuiltinBitOp<cir::BitClzOp>(*this, e, /*poisonZero=*/true);

  case Builtin::BI__builtin_ffs:
  case Builtin::BI__builtin_ffsl:
  case Builtin::BI__builtin_ffsll:
    return emitBuiltinBitOp<cir::BitFfsOp>(*this, e);

  case Builtin::BI__builtin_parity:
  case Builtin::BI__builtin_parityl:
  case Builtin::BI__builtin_parityll:
    return emitBuiltinBitOp<cir::BitParityOp>(*this, e);

  case Builtin::BI__lzcnt16:
  case Builtin::BI__lzcnt:
  case Builtin::BI__lzcnt64:
    assert(!cir::MissingFeatures::builtinCheckKind());
    return emitBuiltinBitOp<cir::BitClzOp>(*this, e, /*poisonZero=*/false);

  case Builtin::BI__popcnt16:
  case Builtin::BI__popcnt:
  case Builtin::BI__popcnt64:
  case Builtin::BI__builtin_popcount:
  case Builtin::BI__builtin_popcountl:
  case Builtin::BI__builtin_popcountll:
  case Builtin::BI__builtin_popcountg:
    return emitBuiltinBitOp<cir::BitPopcountOp>(*this, e);

  case Builtin::BI__builtin_expect:
  case Builtin::BI__builtin_expect_with_probability: {
    mlir::Value argValue = emitScalarExpr(e->getArg(0));
    mlir::Value expectedValue = emitScalarExpr(e->getArg(1));

    mlir::FloatAttr probAttr;
    if (builtinIDIfNoAsmLabel == Builtin::BI__builtin_expect_with_probability) {
      llvm::APFloat probability(0.0);
      const Expr *probArg = e->getArg(2);
      [[maybe_unused]] bool evalSucceeded =
          probArg->EvaluateAsFloat(probability, cgm.getASTContext());
      assert(evalSucceeded &&
             "probability should be able to evaluate as float");
      bool loseInfo = false; // ignored
      probability.convert(llvm::APFloat::IEEEdouble(),
                          llvm::RoundingMode::Dynamic, &loseInfo);
      probAttr = mlir::FloatAttr::get(mlir::Float64Type::get(&getMLIRContext()),
                                      probability);
    }

    auto result = cir::ExpectOp::create(builder, loc, argValue.getType(),
                                        argValue, expectedValue, probAttr);
    return RValue::get(result);
  }

  case Builtin::BI__builtin_bswap16:
  case Builtin::BI__builtin_bswap32:
  case Builtin::BI__builtin_bswap64:
  case Builtin::BI_byteswap_ushort:
  case Builtin::BI_byteswap_ulong:
  case Builtin::BI_byteswap_uint64: {
    mlir::Value arg = emitScalarExpr(e->getArg(0));
    return RValue::get(cir::ByteSwapOp::create(builder, loc, arg));
  }

  case Builtin::BI__builtin_bitreverse8:
  case Builtin::BI__builtin_bitreverse16:
  case Builtin::BI__builtin_bitreverse32:
  case Builtin::BI__builtin_bitreverse64: {
    mlir::Value arg = emitScalarExpr(e->getArg(0));
    return RValue::get(cir::BitReverseOp::create(builder, loc, arg));
  }

  case Builtin::BI__builtin_rotateleft8:
  case Builtin::BI__builtin_rotateleft16:
  case Builtin::BI__builtin_rotateleft32:
  case Builtin::BI__builtin_rotateleft64:
    return emitRotate(e, /*isRotateLeft=*/true);

  case Builtin::BI__builtin_rotateright8:
  case Builtin::BI__builtin_rotateright16:
  case Builtin::BI__builtin_rotateright32:
  case Builtin::BI__builtin_rotateright64:
    return emitRotate(e, /*isRotateLeft=*/false);

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
    cgm.errorNYI(e->getSourceRange(), "BI__builtin_coro_id like NYI");
    return getUndefRValue(e->getType());

  case Builtin::BI__builtin_coro_frame: {
    return emitCoroutineFrame();
  }
  case Builtin::BI__builtin_coro_free:
  case Builtin::BI__builtin_coro_size: {
    GlobalDecl gd{fd};
    mlir::Type ty = cgm.getTypes().getFunctionType(
        cgm.getTypes().arrangeGlobalDeclaration(gd));
    const auto *nd = cast<NamedDecl>(gd.getDecl());
    cir::FuncOp fnOp =
        cgm.getOrCreateCIRFunction(nd->getName(), ty, gd, /*ForVTable=*/false);
    fnOp.setBuiltin(true);
    return emitCall(e->getCallee()->getType(), CIRGenCallee::forDirect(fnOp), e,
                    returnValue);
  }

  case Builtin::BI__builtin_constant_p: {
    mlir::Type resultType = convertType(e->getType());

    const Expr *arg = e->getArg(0);
    QualType argType = arg->getType();
    // FIXME: The allowance for Obj-C pointers and block pointers is historical
    // and likely a mistake.
    if (!argType->isIntegralOrEnumerationType() && !argType->isFloatingType() &&
        !argType->isObjCObjectPointerType() && !argType->isBlockPointerType()) {
      // Per the GCC documentation, only numeric constants are recognized after
      // inlining.
      return RValue::get(
          builder.getConstInt(getLoc(e->getSourceRange()),
                              mlir::cast<cir::IntType>(resultType), 0));
    }

    if (arg->HasSideEffects(getContext())) {
      // The argument is unevaluated, so be conservative if it might have
      // side-effects.
      return RValue::get(
          builder.getConstInt(getLoc(e->getSourceRange()),
                              mlir::cast<cir::IntType>(resultType), 0));
    }

    mlir::Value argValue = emitScalarExpr(arg);
    if (argType->isObjCObjectPointerType()) {
      cgm.errorNYI(e->getSourceRange(),
                   "__builtin_constant_p: Obj-C object pointer");
      return {};
    }
    argValue = builder.createBitcast(argValue, convertType(argType));

    mlir::Value result = cir::IsConstantOp::create(
        builder, getLoc(e->getSourceRange()), argValue);
    // IsConstantOp returns a bool, but __builtin_constant_p returns an int.
    result = builder.createBoolToInt(result, resultType);
    return RValue::get(result);
  }
  case Builtin::BI__builtin_dynamic_object_size:
  case Builtin::BI__builtin_object_size: {
    unsigned type =
        e->getArg(1)->EvaluateKnownConstInt(getContext()).getZExtValue();
    auto resType = mlir::cast<cir::IntType>(convertType(e->getType()));

    // We pass this builtin onto the optimizer so that it can figure out the
    // object size in more complex cases.
    bool isDynamic = builtinID == Builtin::BI__builtin_dynamic_object_size;
    return RValue::get(emitBuiltinObjectSize(e->getArg(0), type, resType,
                                             /*EmittedE=*/nullptr, isDynamic));
  }

  case Builtin::BI__builtin_prefetch: {
    auto evaluateOperandAsInt = [&](const Expr *arg) {
      Expr::EvalResult res;
      [[maybe_unused]] bool evalSucceed =
          arg->EvaluateAsInt(res, cgm.getASTContext());
      assert(evalSucceed && "expression should be able to evaluate as int");
      return res.Val.getInt().getZExtValue();
    };

    bool isWrite = false;
    if (e->getNumArgs() > 1)
      isWrite = evaluateOperandAsInt(e->getArg(1));

    int locality = 3;
    if (e->getNumArgs() > 2)
      locality = evaluateOperandAsInt(e->getArg(2));

    mlir::Value address = emitScalarExpr(e->getArg(0));
    cir::PrefetchOp::create(builder, loc, address, locality, isWrite);
    return RValue::get(nullptr);
  }
  case Builtin::BI__builtin_readcyclecounter:
  case Builtin::BI__builtin_readsteadycounter:
  case Builtin::BI__builtin___clear_cache:
    return errorBuiltinNYI(*this, e, builtinID);
  case Builtin::BI__builtin_trap:
    emitTrap(loc, /*createNewBlock=*/true);
    return RValue::getIgnored();
  case Builtin::BI__builtin_verbose_trap:
  case Builtin::BI__debugbreak:
    return errorBuiltinNYI(*this, e, builtinID);
  case Builtin::BI__builtin_unreachable:
    emitUnreachable(e->getExprLoc(), /*createNewBlock=*/true);
    return RValue::getIgnored();
  case Builtin::BI__builtin_powi:
  case Builtin::BI__builtin_powif:
  case Builtin::BI__builtin_powil:
  case Builtin::BI__builtin_frexpl:
  case Builtin::BI__builtin_frexp:
  case Builtin::BI__builtin_frexpf:
  case Builtin::BI__builtin_frexpf128:
  case Builtin::BI__builtin_frexpf16:
  case Builtin::BImodf:
  case Builtin::BImodff:
  case Builtin::BImodfl:
  case Builtin::BI__builtin_modf:
  case Builtin::BI__builtin_modff:
  case Builtin::BI__builtin_modfl:
  case Builtin::BI__builtin_isgreater:
  case Builtin::BI__builtin_isgreaterequal:
  case Builtin::BI__builtin_isless:
  case Builtin::BI__builtin_islessequal:
  case Builtin::BI__builtin_islessgreater:
  case Builtin::BI__builtin_isunordered:
  // From https://clang.llvm.org/docs/LanguageExtensions.html#builtin-isfpclass
  //
  //  The `__builtin_isfpclass()` builtin is a generalization of functions
  //  isnan, isinf, isfinite and some others defined by the C standard. It tests
  //  if the floating-point value, specified by the first argument, falls into
  //  any of data classes, specified by the second argument.
  case Builtin::BI__builtin_isnan: {
    assert(!cir::MissingFeatures::cgFPOptionsRAII());
    mlir::Value v = emitScalarExpr(e->getArg(0));
    assert(!cir::MissingFeatures::fpConstraints());
    mlir::Location loc = getLoc(e->getBeginLoc());
    return RValue::get(builder.createBoolToInt(
        builder.createIsFPClass(loc, v, cir::FPClassTest::Nan),
        convertType(e->getType())));
  }

  case Builtin::BI__builtin_issignaling: {
    assert(!cir::MissingFeatures::cgFPOptionsRAII());
    mlir::Value v = emitScalarExpr(e->getArg(0));
    mlir::Location loc = getLoc(e->getBeginLoc());
    return RValue::get(builder.createBoolToInt(
        builder.createIsFPClass(loc, v, cir::FPClassTest::SignalingNaN),
        convertType(e->getType())));
  }

  case Builtin::BI__builtin_isinf: {
    assert(!cir::MissingFeatures::cgFPOptionsRAII());
    mlir::Value v = emitScalarExpr(e->getArg(0));
    assert(!cir::MissingFeatures::fpConstraints());
    mlir::Location loc = getLoc(e->getBeginLoc());
    return RValue::get(builder.createBoolToInt(
        builder.createIsFPClass(loc, v, cir::FPClassTest::Infinity),
        convertType(e->getType())));
  }
  case Builtin::BIfinite:
  case Builtin::BI__finite:
  case Builtin::BIfinitef:
  case Builtin::BI__finitef:
  case Builtin::BIfinitel:
  case Builtin::BI__finitel:
  case Builtin::BI__builtin_isfinite: {
    assert(!cir::MissingFeatures::cgFPOptionsRAII());
    mlir::Value v = emitScalarExpr(e->getArg(0));
    assert(!cir::MissingFeatures::fpConstraints());
    mlir::Location loc = getLoc(e->getBeginLoc());
    return RValue::get(builder.createBoolToInt(
        builder.createIsFPClass(loc, v, cir::FPClassTest::Finite),
        convertType(e->getType())));
  }

  case Builtin::BI__builtin_isnormal: {
    assert(!cir::MissingFeatures::cgFPOptionsRAII());
    mlir::Value v = emitScalarExpr(e->getArg(0));
    mlir::Location loc = getLoc(e->getBeginLoc());
    return RValue::get(builder.createBoolToInt(
        builder.createIsFPClass(loc, v, cir::FPClassTest::Normal),
        convertType(e->getType())));
  }

  case Builtin::BI__builtin_issubnormal: {
    assert(!cir::MissingFeatures::cgFPOptionsRAII());
    mlir::Value v = emitScalarExpr(e->getArg(0));
    mlir::Location loc = getLoc(e->getBeginLoc());
    return RValue::get(builder.createBoolToInt(
        builder.createIsFPClass(loc, v, cir::FPClassTest::Subnormal),
        convertType(e->getType())));
  }

  case Builtin::BI__builtin_iszero: {
    assert(!cir::MissingFeatures::cgFPOptionsRAII());
    mlir::Value v = emitScalarExpr(e->getArg(0));
    mlir::Location loc = getLoc(e->getBeginLoc());
    return RValue::get(builder.createBoolToInt(
        builder.createIsFPClass(loc, v, cir::FPClassTest::Zero),
        convertType(e->getType())));
  }
  case Builtin::BI__builtin_isfpclass: {
    Expr::EvalResult result;
    if (!e->getArg(1)->EvaluateAsInt(result, cgm.getASTContext()))
      break;

    assert(!cir::MissingFeatures::cgFPOptionsRAII());
    mlir::Value v = emitScalarExpr(e->getArg(0));
    uint64_t test = result.Val.getInt().getLimitedValue();
    mlir::Location loc = getLoc(e->getBeginLoc());
    //
    return RValue::get(builder.createBoolToInt(
        builder.createIsFPClass(loc, v, cir::FPClassTest(test)),
        convertType(e->getType())));
  }
  case Builtin::BI__builtin_nondeterministic_value:
  case Builtin::BI__builtin_elementwise_abs:
    return errorBuiltinNYI(*this, e, builtinID);
  case Builtin::BI__builtin_elementwise_acos:
    return emitUnaryFPBuiltin<cir::ACosOp>(*this, *e);
  case Builtin::BI__builtin_elementwise_asin:
    return emitUnaryFPBuiltin<cir::ASinOp>(*this, *e);
  case Builtin::BI__builtin_elementwise_atan:
    return emitUnaryFPBuiltin<cir::ATanOp>(*this, *e);
  case Builtin::BI__builtin_elementwise_atan2:
  case Builtin::BI__builtin_elementwise_ceil:
  case Builtin::BI__builtin_elementwise_exp:
  case Builtin::BI__builtin_elementwise_exp2:
  case Builtin::BI__builtin_elementwise_exp10:
  case Builtin::BI__builtin_elementwise_ldexp:
  case Builtin::BI__builtin_elementwise_log:
  case Builtin::BI__builtin_elementwise_log2:
  case Builtin::BI__builtin_elementwise_log10:
  case Builtin::BI__builtin_elementwise_pow:
  case Builtin::BI__builtin_elementwise_bitreverse:
    return errorBuiltinNYI(*this, e, builtinID);
  case Builtin::BI__builtin_elementwise_cos:
    return emitUnaryFPBuiltin<cir::CosOp>(*this, *e);
  case Builtin::BI__builtin_elementwise_cosh:
  case Builtin::BI__builtin_elementwise_floor:
  case Builtin::BI__builtin_elementwise_popcount:
  case Builtin::BI__builtin_elementwise_roundeven:
  case Builtin::BI__builtin_elementwise_round:
  case Builtin::BI__builtin_elementwise_rint:
  case Builtin::BI__builtin_elementwise_nearbyint:
  case Builtin::BI__builtin_elementwise_sin:
  case Builtin::BI__builtin_elementwise_sinh:
  case Builtin::BI__builtin_elementwise_tan:
  case Builtin::BI__builtin_elementwise_tanh:
  case Builtin::BI__builtin_elementwise_trunc:
  case Builtin::BI__builtin_elementwise_canonicalize:
  case Builtin::BI__builtin_elementwise_copysign:
  case Builtin::BI__builtin_elementwise_fma:
  case Builtin::BI__builtin_elementwise_fshl:
  case Builtin::BI__builtin_elementwise_fshr:
  case Builtin::BI__builtin_elementwise_add_sat:
  case Builtin::BI__builtin_elementwise_sub_sat:
  case Builtin::BI__builtin_elementwise_max:
  case Builtin::BI__builtin_elementwise_min:
  case Builtin::BI__builtin_elementwise_maxnum:
  case Builtin::BI__builtin_elementwise_minnum:
  case Builtin::BI__builtin_elementwise_maximum:
  case Builtin::BI__builtin_elementwise_minimum:
  case Builtin::BI__builtin_elementwise_maximumnum:
  case Builtin::BI__builtin_elementwise_minimumnum:
  case Builtin::BI__builtin_reduce_max:
  case Builtin::BI__builtin_reduce_min:
  case Builtin::BI__builtin_reduce_add:
  case Builtin::BI__builtin_reduce_mul:
  case Builtin::BI__builtin_reduce_xor:
  case Builtin::BI__builtin_reduce_or:
  case Builtin::BI__builtin_reduce_and:
  case Builtin::BI__builtin_reduce_maximum:
  case Builtin::BI__builtin_reduce_minimum:
  case Builtin::BI__builtin_matrix_transpose:
  case Builtin::BI__builtin_matrix_column_major_load:
  case Builtin::BI__builtin_matrix_column_major_store:
  case Builtin::BI__builtin_masked_load:
  case Builtin::BI__builtin_masked_expand_load:
  case Builtin::BI__builtin_masked_gather:
  case Builtin::BI__builtin_masked_store:
  case Builtin::BI__builtin_masked_compress_store:
  case Builtin::BI__builtin_masked_scatter:
  case Builtin::BI__builtin_isinf_sign:
  case Builtin::BI__builtin_flt_rounds:
  case Builtin::BI__builtin_set_flt_rounds:
  case Builtin::BI__builtin_fpclassify:
    return errorBuiltinNYI(*this, e, builtinID);
  case Builtin::BIalloca:
  case Builtin::BI_alloca:
  case Builtin::BI__builtin_alloca_uninitialized:
  case Builtin::BI__builtin_alloca:
    return emitBuiltinAlloca(*this, e, builtinID);
  case Builtin::BI__builtin_alloca_with_align_uninitialized:
  case Builtin::BI__builtin_alloca_with_align:
  case Builtin::BI__builtin_infer_alloc_token:
  case Builtin::BIbzero:
  case Builtin::BI__builtin_bzero:
  case Builtin::BIbcopy:
  case Builtin::BI__builtin_bcopy:
    return errorBuiltinNYI(*this, e, builtinID);
  case Builtin::BImemcpy:
  case Builtin::BI__builtin_memcpy:
  case Builtin::BImempcpy:
  case Builtin::BI__builtin_mempcpy:
  case Builtin::BI__builtin_memcpy_inline:
  case Builtin::BI__builtin_char_memchr:
  case Builtin::BI__builtin___memcpy_chk:
  case Builtin::BI__builtin_objc_memmove_collectable:
  case Builtin::BI__builtin___memmove_chk:
  case Builtin::BI__builtin_trivially_relocate:
  case Builtin::BImemmove:
  case Builtin::BI__builtin_memmove:
  case Builtin::BImemset:
  case Builtin::BI__builtin_memset:
  case Builtin::BI__builtin_memset_inline:
  case Builtin::BI__builtin___memset_chk:
  case Builtin::BI__builtin_wmemchr:
  case Builtin::BI__builtin_wmemcmp:
    break; // Handled as library calls below.
  case Builtin::BI__builtin_dwarf_cfa:
    return errorBuiltinNYI(*this, e, builtinID);
  case Builtin::BI__builtin_return_address: {
    llvm::APSInt level = e->getArg(0)->EvaluateKnownConstInt(getContext());
    return RValue::get(cir::ReturnAddrOp::create(
        builder, getLoc(e->getExprLoc()),
        builder.getConstAPInt(loc, builder.getUInt32Ty(), level)));
  }
  case Builtin::BI_ReturnAddress: {
    return RValue::get(cir::ReturnAddrOp::create(
        builder, getLoc(e->getExprLoc()),
        builder.getConstInt(loc, builder.getUInt32Ty(), 0)));
  }
  case Builtin::BI__builtin_frame_address: {
    llvm::APSInt level = e->getArg(0)->EvaluateKnownConstInt(getContext());
    mlir::Location loc = getLoc(e->getExprLoc());
    mlir::Value addr = cir::FrameAddrOp::create(
        builder, loc, allocaInt8PtrTy,
        builder.getConstAPInt(loc, builder.getUInt32Ty(), level));
    return RValue::get(
        builder.createCast(loc, cir::CastKind::bitcast, addr, voidPtrTy));
  }
  case Builtin::BI__builtin_extract_return_addr:
  case Builtin::BI__builtin_frob_return_addr:
  case Builtin::BI__builtin_dwarf_sp_column:
  case Builtin::BI__builtin_init_dwarf_reg_size_table:
  case Builtin::BI__builtin_eh_return:
  case Builtin::BI__builtin_unwind_init:
  case Builtin::BI__builtin_extend_pointer:
  case Builtin::BI__builtin_setjmp:
  case Builtin::BI__builtin_longjmp:
  case Builtin::BI__builtin_launder:
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
  case Builtin::BI__sync_fetch_and_add_1:
  case Builtin::BI__sync_fetch_and_add_2:
  case Builtin::BI__sync_fetch_and_add_4:
  case Builtin::BI__sync_fetch_and_add_8:
  case Builtin::BI__sync_fetch_and_add_16:
  case Builtin::BI__sync_fetch_and_sub_1:
  case Builtin::BI__sync_fetch_and_sub_2:
  case Builtin::BI__sync_fetch_and_sub_4:
  case Builtin::BI__sync_fetch_and_sub_8:
  case Builtin::BI__sync_fetch_and_sub_16:
  case Builtin::BI__sync_fetch_and_or_1:
  case Builtin::BI__sync_fetch_and_or_2:
  case Builtin::BI__sync_fetch_and_or_4:
  case Builtin::BI__sync_fetch_and_or_8:
  case Builtin::BI__sync_fetch_and_or_16:
  case Builtin::BI__sync_fetch_and_and_1:
  case Builtin::BI__sync_fetch_and_and_2:
  case Builtin::BI__sync_fetch_and_and_4:
  case Builtin::BI__sync_fetch_and_and_8:
  case Builtin::BI__sync_fetch_and_and_16:
  case Builtin::BI__sync_fetch_and_xor_1:
  case Builtin::BI__sync_fetch_and_xor_2:
  case Builtin::BI__sync_fetch_and_xor_4:
  case Builtin::BI__sync_fetch_and_xor_8:
  case Builtin::BI__sync_fetch_and_xor_16:
  case Builtin::BI__sync_fetch_and_nand_1:
  case Builtin::BI__sync_fetch_and_nand_2:
  case Builtin::BI__sync_fetch_and_nand_4:
  case Builtin::BI__sync_fetch_and_nand_8:
  case Builtin::BI__sync_fetch_and_nand_16:
  case Builtin::BI__sync_fetch_and_min:
  case Builtin::BI__sync_fetch_and_max:
  case Builtin::BI__sync_fetch_and_umin:
  case Builtin::BI__sync_fetch_and_umax:
    return errorBuiltinNYI(*this, e, builtinID);
    return getUndefRValue(e->getType());
  case Builtin::BI__sync_add_and_fetch_1:
  case Builtin::BI__sync_add_and_fetch_2:
  case Builtin::BI__sync_add_and_fetch_4:
  case Builtin::BI__sync_add_and_fetch_8:
  case Builtin::BI__sync_add_and_fetch_16:
    return emitBinaryAtomicPost(*this, cir::AtomicFetchKind::Add, e,
                                cir::BinOpKind::Add);
  case Builtin::BI__sync_sub_and_fetch_1:
  case Builtin::BI__sync_sub_and_fetch_2:
  case Builtin::BI__sync_sub_and_fetch_4:
  case Builtin::BI__sync_sub_and_fetch_8:
  case Builtin::BI__sync_sub_and_fetch_16:
    return emitBinaryAtomicPost(*this, cir::AtomicFetchKind::Sub, e,
                                cir::BinOpKind::Sub);
  case Builtin::BI__sync_and_and_fetch_1:
  case Builtin::BI__sync_and_and_fetch_2:
  case Builtin::BI__sync_and_and_fetch_4:
  case Builtin::BI__sync_and_and_fetch_8:
  case Builtin::BI__sync_and_and_fetch_16:
    return emitBinaryAtomicPost(*this, cir::AtomicFetchKind::And, e,
                                cir::BinOpKind::And);
  case Builtin::BI__sync_or_and_fetch_1:
  case Builtin::BI__sync_or_and_fetch_2:
  case Builtin::BI__sync_or_and_fetch_4:
  case Builtin::BI__sync_or_and_fetch_8:
  case Builtin::BI__sync_or_and_fetch_16:
    return emitBinaryAtomicPost(*this, cir::AtomicFetchKind::Or, e,
                                cir::BinOpKind::Or);
  case Builtin::BI__sync_xor_and_fetch_1:
  case Builtin::BI__sync_xor_and_fetch_2:
  case Builtin::BI__sync_xor_and_fetch_4:
  case Builtin::BI__sync_xor_and_fetch_8:
  case Builtin::BI__sync_xor_and_fetch_16:
    return emitBinaryAtomicPost(*this, cir::AtomicFetchKind::Xor, e,
                                cir::BinOpKind::Xor);
  case Builtin::BI__sync_nand_and_fetch_1:
  case Builtin::BI__sync_nand_and_fetch_2:
  case Builtin::BI__sync_nand_and_fetch_4:
  case Builtin::BI__sync_nand_and_fetch_8:
  case Builtin::BI__sync_nand_and_fetch_16:
    return emitBinaryAtomicPost(*this, cir::AtomicFetchKind::Nand, e,
                                cir::BinOpKind::And, true);
  case Builtin::BI__sync_val_compare_and_swap_1:
  case Builtin::BI__sync_val_compare_and_swap_2:
  case Builtin::BI__sync_val_compare_and_swap_4:
  case Builtin::BI__sync_val_compare_and_swap_8:
  case Builtin::BI__sync_val_compare_and_swap_16:
  case Builtin::BI__sync_bool_compare_and_swap_1:
  case Builtin::BI__sync_bool_compare_and_swap_2:
  case Builtin::BI__sync_bool_compare_and_swap_4:
  case Builtin::BI__sync_bool_compare_and_swap_8:
  case Builtin::BI__sync_bool_compare_and_swap_16:
  case Builtin::BI__sync_swap_1:
  case Builtin::BI__sync_swap_2:
  case Builtin::BI__sync_swap_4:
  case Builtin::BI__sync_swap_8:
  case Builtin::BI__sync_swap_16:
  case Builtin::BI__sync_lock_test_and_set_1:
  case Builtin::BI__sync_lock_test_and_set_2:
  case Builtin::BI__sync_lock_test_and_set_4:
  case Builtin::BI__sync_lock_test_and_set_8:
  case Builtin::BI__sync_lock_test_and_set_16:
  case Builtin::BI__sync_lock_release_1:
  case Builtin::BI__sync_lock_release_2:
  case Builtin::BI__sync_lock_release_4:
  case Builtin::BI__sync_lock_release_8:
  case Builtin::BI__sync_lock_release_16:
  case Builtin::BI__sync_synchronize:
  case Builtin::BI__builtin_nontemporal_load:
  case Builtin::BI__builtin_nontemporal_store:
  case Builtin::BI__c11_atomic_is_lock_free:
  case Builtin::BI__atomic_is_lock_free:
  case Builtin::BI__atomic_test_and_set:
  case Builtin::BI__atomic_clear:
    return errorBuiltinNYI(*this, e, builtinID);
  case Builtin::BI__atomic_thread_fence:
  case Builtin::BI__c11_atomic_thread_fence: {
    emitAtomicFenceOp(*this, e, cir::SyncScopeKind::System);
    return RValue::get(nullptr);
  }
  case Builtin::BI__atomic_signal_fence:
  case Builtin::BI__c11_atomic_signal_fence: {
    emitAtomicFenceOp(*this, e, cir::SyncScopeKind::SingleThread);
    return RValue::get(nullptr);
  }
  case Builtin::BI__scoped_atomic_thread_fence:
  case Builtin::BI__builtin_signbit:
  case Builtin::BI__builtin_signbitf:
  case Builtin::BI__builtin_signbitl:
  case Builtin::BI__warn_memset_zero_len:
  case Builtin::BI__annotation:
  case Builtin::BI__builtin_annotation:
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
    return errorBuiltinNYI(*this, e, builtinID);

  case Builtin::BI__builtin_add_overflow:
  case Builtin::BI__builtin_sub_overflow:
  case Builtin::BI__builtin_mul_overflow: {
    const clang::Expr *leftArg = e->getArg(0);
    const clang::Expr *rightArg = e->getArg(1);
    const clang::Expr *resultArg = e->getArg(2);

    clang::QualType resultQTy =
        resultArg->getType()->castAs<clang::PointerType>()->getPointeeType();

    WidthAndSignedness leftInfo =
        getIntegerWidthAndSignedness(cgm.getASTContext(), leftArg->getType());
    WidthAndSignedness rightInfo =
        getIntegerWidthAndSignedness(cgm.getASTContext(), rightArg->getType());
    WidthAndSignedness resultInfo =
        getIntegerWidthAndSignedness(cgm.getASTContext(), resultQTy);

    // Note we compute the encompassing type with the consideration to the
    // result type, so later in LLVM lowering we don't get redundant integral
    // extension casts.
    WidthAndSignedness encompassingInfo =
        EncompassingIntegerType({leftInfo, rightInfo, resultInfo});

    auto encompassingCIRTy = cir::IntType::get(
        &getMLIRContext(), encompassingInfo.width, encompassingInfo.isSigned);
    auto resultCIRTy = mlir::cast<cir::IntType>(cgm.convertType(resultQTy));

    mlir::Value left = emitScalarExpr(leftArg);
    mlir::Value right = emitScalarExpr(rightArg);
    Address resultPtr = emitPointerWithAlignment(resultArg);

    // Extend each operand to the encompassing type, if necessary.
    if (left.getType() != encompassingCIRTy)
      left =
          builder.createCast(cir::CastKind::integral, left, encompassingCIRTy);
    if (right.getType() != encompassingCIRTy)
      right =
          builder.createCast(cir::CastKind::integral, right, encompassingCIRTy);

    // Perform the operation on the extended values.
    cir::BinOpOverflowKind opKind;
    switch (builtinID) {
    default:
      llvm_unreachable("Unknown overflow builtin id.");
    case Builtin::BI__builtin_add_overflow:
      opKind = cir::BinOpOverflowKind::Add;
      break;
    case Builtin::BI__builtin_sub_overflow:
      opKind = cir::BinOpOverflowKind::Sub;
      break;
    case Builtin::BI__builtin_mul_overflow:
      opKind = cir::BinOpOverflowKind::Mul;
      break;
    }

    mlir::Location loc = getLoc(e->getSourceRange());
    auto arithOp = cir::BinOpOverflowOp::create(builder, loc, resultCIRTy,
                                                opKind, left, right);

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
        resultArg->getType()->getPointeeType().isVolatileQualified();
    builder.createStore(loc, arithOp.getResult(), resultPtr, isVolatile);

    return RValue::get(arithOp.getOverflow());
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
    mlir::Value x = emitScalarExpr(e->getArg(0));
    mlir::Value y = emitScalarExpr(e->getArg(1));

    const clang::Expr *resultArg = e->getArg(2);
    Address resultPtr = emitPointerWithAlignment(resultArg);

    // Decide which of the arithmetic operation we are lowering to:
    cir::BinOpOverflowKind arithKind;
    switch (builtinID) {
    default:
      llvm_unreachable("Unknown overflow builtin id.");
    case Builtin::BI__builtin_uadd_overflow:
    case Builtin::BI__builtin_uaddl_overflow:
    case Builtin::BI__builtin_uaddll_overflow:
    case Builtin::BI__builtin_sadd_overflow:
    case Builtin::BI__builtin_saddl_overflow:
    case Builtin::BI__builtin_saddll_overflow:
      arithKind = cir::BinOpOverflowKind::Add;
      break;
    case Builtin::BI__builtin_usub_overflow:
    case Builtin::BI__builtin_usubl_overflow:
    case Builtin::BI__builtin_usubll_overflow:
    case Builtin::BI__builtin_ssub_overflow:
    case Builtin::BI__builtin_ssubl_overflow:
    case Builtin::BI__builtin_ssubll_overflow:
      arithKind = cir::BinOpOverflowKind::Sub;
      break;
    case Builtin::BI__builtin_umul_overflow:
    case Builtin::BI__builtin_umull_overflow:
    case Builtin::BI__builtin_umulll_overflow:
    case Builtin::BI__builtin_smul_overflow:
    case Builtin::BI__builtin_smull_overflow:
    case Builtin::BI__builtin_smulll_overflow:
      arithKind = cir::BinOpOverflowKind::Mul;
      break;
    }

    clang::QualType resultQTy =
        resultArg->getType()->castAs<clang::PointerType>()->getPointeeType();
    auto resultCIRTy = mlir::cast<cir::IntType>(cgm.convertType(resultQTy));

    mlir::Location loc = getLoc(e->getSourceRange());
    cir::BinOpOverflowOp arithOp = cir::BinOpOverflowOp::create(
        builder, loc, resultCIRTy, arithKind, x, y);

    bool isVolatile =
        resultArg->getType()->getPointeeType().isVolatileQualified();
    builder.createStore(loc, emitToMemory(arithOp.getResult(), resultQTy),
                        resultPtr, isVolatile);

    return RValue::get(arithOp.getOverflow());
  }

  case Builtin::BIaddressof:
  case Builtin::BI__addressof:
  case Builtin::BI__builtin_addressof:
  case Builtin::BI__builtin_function_start:
    return errorBuiltinNYI(*this, e, builtinID);
  case Builtin::BI__builtin_operator_new:
    return emitNewOrDeleteBuiltinCall(
        e->getCallee()->getType()->castAs<FunctionProtoType>(), e, OO_New);
  case Builtin::BI__builtin_operator_delete:
    emitNewOrDeleteBuiltinCall(
        e->getCallee()->getType()->castAs<FunctionProtoType>(), e, OO_Delete);
    return RValue::get(nullptr);
  case Builtin::BI__builtin_is_aligned:
  case Builtin::BI__builtin_align_up:
  case Builtin::BI__builtin_align_down:
  case Builtin::BI__noop:
  case Builtin::BI__builtin_call_with_static_chain:
  case Builtin::BI_InterlockedExchange8:
  case Builtin::BI_InterlockedExchange16:
  case Builtin::BI_InterlockedExchange:
  case Builtin::BI_InterlockedExchangePointer:
  case Builtin::BI_InterlockedCompareExchangePointer:
  case Builtin::BI_InterlockedCompareExchangePointer_nf:
  case Builtin::BI_InterlockedCompareExchange8:
  case Builtin::BI_InterlockedCompareExchange16:
  case Builtin::BI_InterlockedCompareExchange:
  case Builtin::BI_InterlockedCompareExchange64:
  case Builtin::BI_InterlockedIncrement16:
  case Builtin::BI_InterlockedIncrement:
  case Builtin::BI_InterlockedDecrement16:
  case Builtin::BI_InterlockedDecrement:
  case Builtin::BI_InterlockedAnd8:
  case Builtin::BI_InterlockedAnd16:
  case Builtin::BI_InterlockedAnd:
  case Builtin::BI_InterlockedExchangeAdd8:
  case Builtin::BI_InterlockedExchangeAdd16:
  case Builtin::BI_InterlockedExchangeAdd:
  case Builtin::BI_InterlockedExchangeSub8:
  case Builtin::BI_InterlockedExchangeSub16:
  case Builtin::BI_InterlockedExchangeSub:
  case Builtin::BI_InterlockedOr8:
  case Builtin::BI_InterlockedOr16:
  case Builtin::BI_InterlockedOr:
  case Builtin::BI_InterlockedXor8:
  case Builtin::BI_InterlockedXor16:
  case Builtin::BI_InterlockedXor:
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
  case Builtin::BI_interlockedbittestandreset64_acq:
  case Builtin::BI_interlockedbittestandreset64_rel:
  case Builtin::BI_interlockedbittestandreset64_nf:
  case Builtin::BI_interlockedbittestandset64:
  case Builtin::BI_interlockedbittestandset64_acq:
  case Builtin::BI_interlockedbittestandset64_rel:
  case Builtin::BI_interlockedbittestandset64_nf:
  case Builtin::BI_interlockedbittestandset:
  case Builtin::BI_interlockedbittestandset_acq:
  case Builtin::BI_interlockedbittestandset_rel:
  case Builtin::BI_interlockedbittestandset_nf:
  case Builtin::BI_interlockedbittestandreset_acq:
  case Builtin::BI_interlockedbittestandreset_rel:
  case Builtin::BI_interlockedbittestandreset_nf:
  case Builtin::BI__iso_volatile_load8:
  case Builtin::BI__iso_volatile_load16:
  case Builtin::BI__iso_volatile_load32:
  case Builtin::BI__iso_volatile_load64:
  case Builtin::BI__iso_volatile_store8:
  case Builtin::BI__iso_volatile_store16:
  case Builtin::BI__iso_volatile_store32:
  case Builtin::BI__iso_volatile_store64:
  case Builtin::BI__builtin_ptrauth_sign_constant:
  case Builtin::BI__builtin_ptrauth_auth:
  case Builtin::BI__builtin_ptrauth_auth_and_resign:
  case Builtin::BI__builtin_ptrauth_blend_discriminator:
  case Builtin::BI__builtin_ptrauth_sign_generic_data:
  case Builtin::BI__builtin_ptrauth_sign_unauthenticated:
  case Builtin::BI__builtin_ptrauth_strip:
  case Builtin::BI__builtin_get_vtable_pointer:
  case Builtin::BI__exception_code:
  case Builtin::BI_exception_code:
  case Builtin::BI__exception_info:
  case Builtin::BI_exception_info:
  case Builtin::BI__abnormal_termination:
  case Builtin::BI_abnormal_termination:
  case Builtin::BI_setjmpex:
  case Builtin::BI_setjmp:
  case Builtin::BImove:
  case Builtin::BImove_if_noexcept:
  case Builtin::BIforward:
  case Builtin::BIforward_like:
  case Builtin::BIas_const:
  case Builtin::BI__GetExceptionInfo:
  case Builtin::BI__fastfail:
  case Builtin::BIread_pipe:
  case Builtin::BIwrite_pipe:
  case Builtin::BIreserve_read_pipe:
  case Builtin::BIreserve_write_pipe:
  case Builtin::BIwork_group_reserve_read_pipe:
  case Builtin::BIwork_group_reserve_write_pipe:
  case Builtin::BIsub_group_reserve_read_pipe:
  case Builtin::BIsub_group_reserve_write_pipe:
  case Builtin::BIcommit_read_pipe:
  case Builtin::BIcommit_write_pipe:
  case Builtin::BIwork_group_commit_read_pipe:
  case Builtin::BIwork_group_commit_write_pipe:
  case Builtin::BIsub_group_commit_read_pipe:
  case Builtin::BIsub_group_commit_write_pipe:
  case Builtin::BIget_pipe_num_packets:
  case Builtin::BIget_pipe_max_packets:
  case Builtin::BIto_global:
  case Builtin::BIto_local:
  case Builtin::BIto_private:
  case Builtin::BIenqueue_kernel:
  case Builtin::BIget_kernel_work_group_size:
  case Builtin::BIget_kernel_preferred_work_group_size_multiple:
  case Builtin::BIget_kernel_max_sub_group_size_for_ndrange:
  case Builtin::BIget_kernel_sub_group_count_for_ndrange:
  case Builtin::BI__builtin_store_half:
  case Builtin::BI__builtin_store_halff:
  case Builtin::BI__builtin_load_half:
  case Builtin::BI__builtin_load_halff:
    return errorBuiltinNYI(*this, e, builtinID);
  case Builtin::BI__builtin_printf:
  case Builtin::BIprintf:
    break;
  case Builtin::BI__builtin_canonicalize:
  case Builtin::BI__builtin_canonicalizef:
  case Builtin::BI__builtin_canonicalizef16:
  case Builtin::BI__builtin_canonicalizel:
  case Builtin::BI__builtin_thread_pointer:
  case Builtin::BI__builtin_os_log_format:
  case Builtin::BI__xray_customevent:
  case Builtin::BI__xray_typedevent:
  case Builtin::BI__builtin_ms_va_start:
  case Builtin::BI__builtin_ms_va_end:
  case Builtin::BI__builtin_ms_va_copy:
  case Builtin::BI__builtin_get_device_side_mangled_name:
    return errorBuiltinNYI(*this, e, builtinID);
  }

  // If this is an alias for a lib function (e.g. __builtin_sin), emit
  // the call using the normal call path, but using the unmangled
  // version of the function name.
  if (getContext().BuiltinInfo.isLibFunction(builtinID))
    return emitLibraryCall(*this, fd, e,
                           cgm.getBuiltinLibFunction(fd, builtinID));

  // Some target-specific builtins can have aggregate return values, e.g.
  // __builtin_arm_mve_vld2q_u32. So if the result is an aggregate, force
  // returnValue to be non-null, so that the target-specific emission code can
  // always just emit into it.
  cir::TypeEvaluationKind evalKind = getEvaluationKind(e->getType());
  if (evalKind == cir::TEK_Aggregate && returnValue.isNull()) {
    cgm.errorNYI(e->getSourceRange(), "aggregate return value from builtin");
    return getUndefRValue(e->getType());
  }

  // Now see if we can emit a target-specific builtin.
  // FIXME: This is a temporary mechanism (double-optional semantics) that will
  // go away once everything is implemented:
  //   1. return `mlir::Value{}` for cases where we have issued the diagnostic.
  //   2. return `std::nullopt` in cases where we didn't issue a diagnostic
  //      but also didn't handle the builtin.
  if (std::optional<mlir::Value> rst =
          emitTargetBuiltinExpr(builtinID, e, returnValue)) {
    mlir::Value v = rst.value();
    // CIR dialect operations may have no results, no values will be returned
    // even if it executes successfully.
    if (!v)
      return RValue::get(nullptr);

    switch (evalKind) {
    case cir::TEK_Scalar:
      if (mlir::isa<cir::VoidType>(v.getType()))
        return RValue::get(nullptr);
      return RValue::get(v);
    case cir::TEK_Aggregate:
      cgm.errorNYI(e->getSourceRange(), "aggregate return value from builtin");
      return getUndefRValue(e->getType());
    case cir::TEK_Complex:
      llvm_unreachable("No current target builtin returns complex");
    }
    llvm_unreachable("Bad evaluation kind in EmitBuiltinExpr");
  }

  cgm.errorNYI(e->getSourceRange(),
               std::string("unimplemented builtin call: ") +
                   getContext().BuiltinInfo.getName(builtinID));
  return getUndefRValue(e->getType());
}

static std::optional<mlir::Value>
emitTargetArchBuiltinExpr(CIRGenFunction *cgf, unsigned builtinID,
                          const CallExpr *e, ReturnValueSlot &returnValue,
                          llvm::Triple::ArchType arch) {
  // When compiling in HipStdPar mode we have to be conservative in rejecting
  // target specific features in the FE, and defer the possible error to the
  // AcceleratorCodeSelection pass, wherein iff an unsupported target builtin is
  // referenced by an accelerator executable function, we emit an error.
  // Returning nullptr here leads to the builtin being handled in
  // EmitStdParUnsupportedBuiltin.
  if (cgf->getLangOpts().HIPStdPar && cgf->getLangOpts().CUDAIsDevice &&
      arch != cgf->getTarget().getTriple().getArch())
    return std::nullopt;

  switch (arch) {
  case llvm::Triple::arm:
  case llvm::Triple::armeb:
  case llvm::Triple::thumb:
  case llvm::Triple::thumbeb:
    // These are actually NYI, but that will be reported by emitBuiltinExpr.
    // At this point, we don't even know that the builtin is target-specific.
    return std::nullopt;
  case llvm::Triple::aarch64:
  case llvm::Triple::aarch64_32:
  case llvm::Triple::aarch64_be:
    return cgf->emitAArch64BuiltinExpr(builtinID, e, returnValue, arch);
  case llvm::Triple::bpfeb:
  case llvm::Triple::bpfel:
    // These are actually NYI, but that will be reported by emitBuiltinExpr.
    // At this point, we don't even know that the builtin is target-specific.
    return std::nullopt;

  case llvm::Triple::x86:
  case llvm::Triple::x86_64:
    return cgf->emitX86BuiltinExpr(builtinID, e);

  case llvm::Triple::ppc:
  case llvm::Triple::ppcle:
  case llvm::Triple::ppc64:
  case llvm::Triple::ppc64le:
  case llvm::Triple::r600:
  case llvm::Triple::amdgcn:
  case llvm::Triple::systemz:
  case llvm::Triple::nvptx:
  case llvm::Triple::nvptx64:
  case llvm::Triple::wasm32:
  case llvm::Triple::wasm64:
  case llvm::Triple::hexagon:
  case llvm::Triple::riscv32:
  case llvm::Triple::riscv64:
    // These are actually NYI, but that will be reported by emitBuiltinExpr.
    // At this point, we don't even know that the builtin is target-specific.
    return std::nullopt;
  default:
    return std::nullopt;
  }
}

std::optional<mlir::Value>
CIRGenFunction::emitTargetBuiltinExpr(unsigned builtinID, const CallExpr *e,
                                      ReturnValueSlot &returnValue) {
  if (getContext().BuiltinInfo.isAuxBuiltinID(builtinID)) {
    assert(getContext().getAuxTargetInfo() && "Missing aux target info");
    return emitTargetArchBuiltinExpr(
        this, getContext().BuiltinInfo.getAuxBuiltinID(builtinID), e,
        returnValue, getContext().getAuxTargetInfo()->getTriple().getArch());
  }

  return emitTargetArchBuiltinExpr(this, builtinID, e, returnValue,
                                   getTarget().getTriple().getArch());
}

mlir::Value CIRGenFunction::emitScalarOrConstFoldImmArg(
    const unsigned iceArguments, const unsigned idx, const Expr *argExpr) {
  mlir::Value arg = {};
  if ((iceArguments & (1 << idx)) == 0) {
    arg = emitScalarExpr(argExpr);
  } else {
    // If this is required to be a constant, constant fold it so that we
    // know that the generated intrinsic gets a ConstantInt.
    const std::optional<llvm::APSInt> result =
        argExpr->getIntegerConstantExpr(getContext());
    assert(result && "Expected argument to be a constant");
    arg = builder.getConstInt(getLoc(argExpr->getSourceRange()), *result);
  }
  return arg;
}

/// Given a builtin id for a function like "__builtin_fabsf", return a Function*
/// for "fabsf".
cir::FuncOp CIRGenModule::getBuiltinLibFunction(const FunctionDecl *fd,
                                                unsigned builtinID) {
  assert(astContext.BuiltinInfo.isLibFunction(builtinID));

  // Get the name, skip over the __builtin_ prefix (if necessary). We may have
  // to build this up so provide a small stack buffer to handle the vast
  // majority of names.
  llvm::SmallString<64> name;

  assert(!cir::MissingFeatures::asmLabelAttr());
  name = astContext.BuiltinInfo.getName(builtinID).substr(10);

  GlobalDecl d(fd);
  mlir::Type type = convertType(fd->getType());
  return getOrCreateCIRFunction(name, type, d, /*forVTable=*/false);
}

mlir::Value CIRGenFunction::emitCheckedArgForAssume(const Expr *e) {
  mlir::Value argValue = evaluateExprAsBool(e);
  if (!sanOpts.has(SanitizerKind::Builtin))
    return argValue;

  assert(!cir::MissingFeatures::sanitizers());
  cgm.errorNYI(e->getSourceRange(),
               "emitCheckedArgForAssume: sanitizers are NYI");
  return {};
}

void CIRGenFunction::emitVAStart(mlir::Value vaList, mlir::Value count) {
  // LLVM codegen casts to *i8, no real gain on doing this for CIRGen this
  // early, defer to LLVM lowering.
  cir::VAStartOp::create(builder, vaList.getLoc(), vaList, count);
}

void CIRGenFunction::emitVAEnd(mlir::Value vaList) {
  cir::VAEndOp::create(builder, vaList.getLoc(), vaList);
}

// FIXME(cir): This completely abstracts away the ABI with a generic CIR Op. By
// default this lowers to llvm.va_arg which is incomplete and not ABI-compliant
// on most targets so cir.va_arg will need some ABI handling in LoweringPrepare
mlir::Value CIRGenFunction::emitVAArg(VAArgExpr *ve) {
  assert(!cir::MissingFeatures::msabi());
  assert(!cir::MissingFeatures::vlas());
  mlir::Location loc = cgm.getLoc(ve->getExprLoc());
  mlir::Type type = convertType(ve->getType());
  mlir::Value vaList = emitVAListRef(ve->getSubExpr()).getPointer();
  return cir::VAArgOp::create(builder, loc, type, vaList);
}

mlir::Value CIRGenFunction::emitBuiltinObjectSize(const Expr *e, unsigned type,
                                                  cir::IntType resType,
                                                  mlir::Value emittedE,
                                                  bool isDynamic) {
  assert(!cir::MissingFeatures::opCallImplicitObjectSizeArgs());

  // LLVM can't handle type=3 appropriately, and __builtin_object_size shouldn't
  // evaluate e for side-effects. In either case, just like original LLVM
  // lowering, we shouldn't lower to `cir.objsize` but to a constant instead.
  if (type == 3 || (!emittedE && e->HasSideEffects(getContext())))
    return builder.getConstInt(getLoc(e->getSourceRange()), resType,
                               (type & 2) ? 0 : -1);

  mlir::Value ptr = emittedE ? emittedE : emitScalarExpr(e);
  assert(mlir::isa<cir::PointerType>(ptr.getType()) &&
         "Non-pointer passed to __builtin_object_size?");

  assert(!cir::MissingFeatures::countedBySize());

  // Extract the min/max mode from type. CIR only supports type 0
  // (max, whole object) and type 2 (min, whole object), not type 1 or 3
  // (closest subobject variants).
  const bool min = ((type & 2) != 0);
  // For GCC compatibility, __builtin_object_size treats NULL as unknown size.
  auto op =
      cir::ObjSizeOp::create(builder, getLoc(e->getSourceRange()), resType, ptr,
                             min, /*nullUnknown=*/true, isDynamic);
  return op.getResult();
}

mlir::Value CIRGenFunction::evaluateOrEmitBuiltinObjectSize(
    const Expr *e, unsigned type, cir::IntType resType, mlir::Value emittedE,
    bool isDynamic) {
  uint64_t objectSize;
  if (!e->tryEvaluateObjectSize(objectSize, getContext(), type))
    return emitBuiltinObjectSize(e, type, resType, emittedE, isDynamic);
  return builder.getConstInt(getLoc(e->getSourceRange()), resType, objectSize);
}

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
#include "clang/AST/Expr.h"
#include "clang/CIR/Dialect/IR/CIRAttrs.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "clang/CIR/MissingFeatures.h"
#include "llvm/IR/Intrinsics.h"

#include "clang/AST/GlobalDecl.h"
#include "clang/Basic/Builtins.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "llvm/Support/ErrorHandling.h"

using namespace clang;
using namespace clang::CIRGen;
using namespace cir;
using namespace llvm;

static RValue emitLibraryCall(CIRGenFunction &cgf, const FunctionDecl *fd,
                              const CallExpr *e, mlir::Operation *calleeValue) {
  CIRGenCallee callee = CIRGenCallee::forDirect(calleeValue, GlobalDecl(fd));
  return cgf.emitCall(e->getCallee()->getType(), callee, e, ReturnValueSlot());
}

static mlir::Type
decodeFixedType(CIRGenFunction &cgf,
                ArrayRef<llvm::Intrinsic::IITDescriptor> &infos) {
  using namespace llvm::Intrinsic;

  auto *context = &cgf.getMLIRContext();
  IITDescriptor descriptor = infos.front();
  infos = infos.slice(1);

  switch (descriptor.Kind) {
  case IITDescriptor::Void:
    return VoidType::get(context);
  case IITDescriptor::Integer:
    return IntType::get(context, descriptor.Integer_Width, /*isSigned=*/true);
  case IITDescriptor::Float:
    return SingleType::get(context);
  case IITDescriptor::Double:
    return DoubleType::get(context);
  default:
    cgf.cgm.errorNYI("intrinsic return types");
    return VoidType::get(context);
  }
}

// llvm::Intrinsics accepts only LLVMContext. We need to reimplement it here.
static cir::FuncType getIntrinsicType(CIRGenFunction &cgf,
                                      llvm::Intrinsic::ID id) {
  using namespace llvm::Intrinsic;

  SmallVector<IITDescriptor, 8> table;
  getIntrinsicInfoTableEntries(id, table);

  ArrayRef<IITDescriptor> tableRef = table;
  mlir::Type resultTy = decodeFixedType(cgf, tableRef);

  SmallVector<mlir::Type, 8> argTypes;
  while (!tableRef.empty())
    argTypes.push_back(decodeFixedType(cgf, tableRef));

  return FuncType::get(argTypes, resultTy);
}

static mlir::Value emitTargetArchBuiltinExpr(CIRGenFunction *cgf,
                                             unsigned builtinID,
                                             const CallExpr *e,
                                             ReturnValueSlot returnValue,
                                             llvm::Triple::ArchType arch) {
  return {};
}

mlir::Value CIRGenFunction::emitTargetBuiltinExpr(unsigned builtinID,
                                                  const CallExpr *e,
                                                  ReturnValueSlot returnValue) {
  if (getContext().BuiltinInfo.isAuxBuiltinID(builtinID)) {
    assert(getContext().getAuxTargetInfo() && "Missing aux target info");
    return emitTargetArchBuiltinExpr(
        this, getContext().BuiltinInfo.getAuxBuiltinID(builtinID), e,
        returnValue, getContext().getAuxTargetInfo()->getTriple().getArch());
  }

  return emitTargetArchBuiltinExpr(this, builtinID, e, returnValue,
                                   getTarget().getTriple().getArch());
}

mlir::Value CIRGenFunction::emitScalarOrConstFoldImmArg(unsigned iceArguments,
                                                        unsigned idx,
                                                        const CallExpr *e) {
  mlir::Value arg = {};
  if ((iceArguments & (1 << idx)) == 0) {
    arg = emitScalarExpr(e->getArg(idx));
  } else {
    // If this is required to be a constant, constant fold it so that we
    // know that the generated intrinsic gets a ConstantInt.
    std::optional<llvm::APSInt> result =
        e->getArg(idx)->getIntegerConstantExpr(getContext());
    assert(result && "Expected argument to be a constant");
    arg = builder.getConstInt(getLoc(e->getSourceRange()), *result);
  }
  return arg;
}

RValue CIRGenFunction::emitBuiltinExpr(const GlobalDecl &gd, unsigned builtinID,
                                       const CallExpr *e,
                                       ReturnValueSlot returnValue) {
  const FunctionDecl *fd = gd.getDecl()->getAsFunction();

  // See if we can constant fold this builtin.  If so, don't emit it at all.
  // TODO: Extend this handling to all builtin calls that we can constant-fold.
  Expr::EvalResult result;
  if (e->isPRValue() && e->EvaluateAsRValue(result, cgm.getASTContext()) &&
      !result.hasSideEffects()) {
    if (result.Val.isInt()) {
      return RValue::get(builder.getConstInt(getLoc(e->getSourceRange()),
                                             result.Val.getInt()));
    }
    if (result.Val.isFloat()) {
      // Note: we are using result type of CallExpr to determine the type of
      // the constant. Clang Codegen uses the result value to make judgement
      // of the type. We feel it should be Ok to use expression type because
      // it is hard to imagine a builtin function evaluates to
      // a value that over/underflows its own defined type.
      mlir::Type resTy = convertType(e->getType());
      return RValue::get(builder.getConstFP(getLoc(e->getExprLoc()), resTy,
                                            result.Val.getFloat()));
    }
  }

  // If current long-double semantics is IEEE 128-bit, replace math builtins
  // of long-double with f128 equivalent.
  // TODO: This mutation should also be applied to other targets other than PPC,
  // after backend supports IEEE 128-bit style libcalls.
  if (getTarget().getTriple().isPPC64() &&
      &getTarget().getLongDoubleFormat() == &llvm::APFloat::IEEEquad()) {
    cgm.errorNYI("long double builtin mutation");
  }

  // If the builtin has been declared explicitly with an assembler label,
  // disable the specialized emitting below. Ideally we should communicate the
  // rename in IR, or at least avoid generating the intrinsic calls that are
  // likely to get lowered to the renamed library functions.
  const unsigned builtinIDIfNoAsmLabel =
      fd->hasAttr<AsmLabelAttr>() ? 0 : builtinID;

  std::optional<bool> errnoOverriden;
  // ErrnoOverriden is true if math-errno is overriden via the
  // '#pragma float_control(precise, on)'. This pragma disables fast-math,
  // which implies math-errno.
  if (e->hasStoredFPFeatures()) {
    FPOptionsOverride op = e->getFPFeatures();
    if (op.hasMathErrnoOverride())
      errnoOverriden = op.getMathErrnoOverride();
  }
  // True if 'atttibute__((optnone)) is used. This attibute overrides
  // fast-math which implies math-errno.
  bool optNone = curFuncDecl && curFuncDecl->hasAttr<OptimizeNoneAttr>();

  // True if we are compiling at -O2 and errno has been disabled
  // using the '#pragma float_control(precise, off)', and
  // attribute opt-none hasn't been seen.
  [[maybe_unused]] bool errnoOverridenToFalseWithOpt =
      errnoOverriden.has_value() && !errnoOverriden.value() && !optNone &&
      cgm.getCodeGenOpts().OptimizationLevel != 0;

  // There are LLVM math intrinsics/instructions corresponding to math library
  // functions except the LLVM op will never set errno while the math library
  // might. Also, math builtins have the same semantics as their math library
  // twins. Thus, we can transform math library and builtin calls to their
  // LLVM counterparts if the call is marked 'const' (known to never set errno).
  // In case FP exceptions are enabled, the experimental versions of the
  // intrinsics model those.
  [[maybe_unused]] bool constAlways =
      getContext().BuiltinInfo.isConst(builtinID);

  // There's a special case with the fma builtins where they are always const
  // if the target environment is GNU or the target is OS is Windows and we're
  // targeting the MSVCRT.dll environment.
  // FIXME: This list can be become outdated. Need to find a way to get it some
  // other way.
  switch (builtinID) {
  case Builtin::BI__builtin_fma:
  case Builtin::BI__builtin_fmaf:
  case Builtin::BI__builtin_fmal:
  case Builtin::BIfma:
  case Builtin::BIfmaf:
  case Builtin::BIfmal:
    cgm.errorNYI("FMA builtins");
    break;
  }

  bool constWithoutErrnoAndExceptions =
      getContext().BuiltinInfo.isConstWithoutErrnoAndExceptions(builtinID);
  bool constWithoutExceptions =
      getContext().BuiltinInfo.isConstWithoutExceptions(builtinID);

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

  bool constWithoutErrnoOrExceptions =
      constWithoutErrnoAndExceptions || constWithoutExceptions;
  bool generateIntrinsics =
      (constAlways && !optNone) ||
      (!getLangOpts().MathErrno &&
       !(errnoOverriden.has_value() && errnoOverriden.value()) && !optNone);
  if (!generateIntrinsics) {
    generateIntrinsics =
        constWithoutErrnoOrExceptions && !constWithoutErrnoAndExceptions;
    if (!generateIntrinsics)
      generateIntrinsics =
          constWithoutErrnoOrExceptions &&
          (!getLangOpts().MathErrno &&
           !(errnoOverriden.has_value() && errnoOverriden.value()) && !optNone);
    if (!generateIntrinsics)
      generateIntrinsics =
          constWithoutErrnoOrExceptions && errnoOverridenToFalseWithOpt;
  }

  if (generateIntrinsics) {
    assert(!cir::MissingFeatures::intrinsics());
    return {};
  }

  switch (builtinIDIfNoAsmLabel) {
  default:
    break;
  }

  // If this is an alias for a lib function (e.g. __builtin_sin), emit
  // the call using the normal call path, but using the unmangled
  // version of the function name.
  if (getContext().BuiltinInfo.isLibFunction(builtinID))
    return emitLibraryCall(*this, fd, e,
                           cgm.getBuiltinLibFunction(fd, builtinID));

  // If this is a predefined lib function (e.g. malloc), emit the call
  // using exactly the normal call path.
  if (getContext().BuiltinInfo.isPredefinedLibFunction(builtinID))
    return emitLibraryCall(*this, fd, e,
                           emitScalarExpr(e->getCallee()).getDefiningOp());

  // Check that a call to a target specific builtin has the correct target
  // features.
  // This is down here to avoid non-target specific builtins, however, if
  // generic builtins start to require generic target features then we
  // can move this up to the beginning of the function.
  //   checkTargetFeatures(E, FD);

  if ([[maybe_unused]] unsigned vectorWidth =
          getContext().BuiltinInfo.getRequiredVectorWidth(builtinID))
    largestVectorWidth = std::max(largestVectorWidth, vectorWidth);

  // See if we have a target specific intrinsic.
  std::string name = getContext().BuiltinInfo.getName(builtinID);
  Intrinsic::ID intrinsicID = Intrinsic::not_intrinsic;
  StringRef prefix =
      llvm::Triple::getArchTypePrefix(getTarget().getTriple().getArch());
  if (!prefix.empty()) {
    intrinsicID = Intrinsic::getIntrinsicForClangBuiltin(prefix.data(), name);
    // NOTE we don't need to perform a compatibility flag check here since the
    // intrinsics are declared in Builtins*.def via LANGBUILTIN which filter the
    // MS builtins via ALL_MS_LANGUAGES and are filtered earlier.
    if (intrinsicID == Intrinsic::not_intrinsic)
      intrinsicID = Intrinsic::getIntrinsicForMSBuiltin(prefix.data(), name);
  }

  if (intrinsicID != Intrinsic::not_intrinsic) {
    unsigned iceArguments = 0;
    ASTContext::GetBuiltinTypeError error;
    getContext().GetBuiltinType(builtinID, error, &iceArguments);
    assert(error == ASTContext::GE_None && "Should not codegen an error");

    llvm::StringRef name = llvm::Intrinsic::getName(intrinsicID);
    // cir::LLVMIntrinsicCallOp expects intrinsic name to not have prefix
    // "llvm." For example, `llvm.nvvm.barrier0` should be passed as
    // `nvvm.barrier0`.
    if (!name.consume_front("llvm."))
      assert(false && "bad intrinsic name!");

    cir::FuncType intrinsicType = getIntrinsicType(*this, intrinsicID);

    SmallVector<mlir::Value> args;
    for (unsigned i = 0; i < e->getNumArgs(); i++) {
      mlir::Value arg = emitScalarOrConstFoldImmArg(iceArguments, i, e);
      mlir::Type argType = arg.getType();
      if (argType != intrinsicType.getInput(i)) {
        //  vector of pointers?
        assert(!cir::MissingFeatures::addressSpace());
      }

      args.push_back(arg);
    }

    auto intrinsicCall = builder.create<cir::LLVMIntrinsicCallOp>(
        getLoc(e->getExprLoc()), builder.getStringAttr(name),
        intrinsicType.getReturnType(), args);

    mlir::Type builtinReturnType = intrinsicCall.getResult().getType();
    mlir::Type retTy = intrinsicType.getReturnType();

    if (builtinReturnType != retTy) {
      // vector of pointers?
      if (isa<cir::PointerType>(retTy)) {
        assert(!cir::MissingFeatures::addressSpace());
      }
    }

    if (isa<cir::VoidType>(retTy))
      return RValue::get(nullptr);

    return RValue::get(intrinsicCall.getResult());
  }

  // Some target-specific builtins can have aggregate return values, e.g.
  // __builtin_arm_mve_vld2q_u32. So if the result is an aggregate, force
  // ReturnValue to be non-null, so that the target-specific emission code can
  // always just emit into it.
  cir::TypeEvaluationKind evalKind = getEvaluationKind(e->getType());
  if (evalKind == cir::TEK_Aggregate && returnValue.isNull()) {
    Address destPtr =
        createMemTemp(e->getType(), getLoc(e->getSourceRange()), "agg.tmp");
    returnValue = ReturnValueSlot(destPtr, false);
  }

  // Now see if we can emit a target-specific builtin.
  if (auto v = emitTargetBuiltinExpr(builtinID, e, returnValue)) {
    switch (evalKind) {
    case cir::TEK_Scalar:
      if (mlir::isa<cir::VoidType>(v.getType()))
        return RValue::get(nullptr);
      return RValue::get(v);
    case cir::TEK_Aggregate:
      return RValue::getAggregate(returnValue.getAddress(),
                                  returnValue.isVolatile());
    case cir::TEK_Complex:
      llvm_unreachable("No current target builtin returns complex");
    }
    llvm_unreachable("Bad evaluation kind in EmitBuiltinExpr");
  }

  // cgm.errorUnsupported(e, "builtin function");

  // Unknown builtin, for now just dump it out and return undef.
  return getUndefRValue(e->getType());
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
  GlobalDecl d(fd);

  // TODO: This list should be expanded or refactored after all GCC-compatible
  // std libcall builtins are implemented.
  static SmallDenseMap<unsigned, StringRef, 64> f128Builtins{
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
  static SmallDenseMap<unsigned, StringRef, 4> aixLongDouble64Builtins{
      {Builtin::BI__builtin_frexpl, "frexp"},
      {Builtin::BI__builtin_ldexpl, "ldexp"},
      {Builtin::BI__builtin_modfl, "modf"},
  };

  // If the builtin has been declared explicitly with an assembler label,
  // use the mangled name. This differs from the plain label on platforms
  // that prefix labels.
  if (fd->hasAttr<AsmLabelAttr>())
    name = getMangledName(d);
  else {
    // TODO: This mutation should also be applied to other targets other than
    // PPC, after backend supports IEEE 128-bit style libcalls.
    if (getTriple().isPPC64() &&
        &getTarget().getLongDoubleFormat() == &llvm::APFloat::IEEEquad() &&
        f128Builtins.find(builtinID) != f128Builtins.end())
      name = f128Builtins[builtinID];
    else if (getTriple().isOSAIX() &&
             &getTarget().getLongDoubleFormat() ==
                 &llvm::APFloat::IEEEdouble() &&
             aixLongDouble64Builtins.find(builtinID) !=
                 aixLongDouble64Builtins.end())
      name = aixLongDouble64Builtins[builtinID];
    else
      name = astContext.BuiltinInfo.getName(builtinID).substr(10);
  }

  auto ty = convertType(fd->getType());
  return getOrCreateCIRFunction(name, ty, d, /*ForVTable=*/false);
}

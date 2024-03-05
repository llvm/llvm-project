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
#include "CIRGenFunction.h"
#include "CIRGenModule.h"
#include "UnimplementedFeatureGuarding.h"

// TODO(cir): we shouldn't need this but we currently reuse intrinsic IDs for
// convenience.
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "llvm/IR/Intrinsics.h"

#include "clang/AST/GlobalDecl.h"
#include "clang/Basic/Builtins.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Value.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "llvm/Support/ErrorHandling.h"

using namespace cir;
using namespace clang;
using namespace mlir::cir;
using namespace llvm;

static RValue buildLibraryCall(CIRGenFunction &CGF, const FunctionDecl *FD,
                               const CallExpr *E,
                               mlir::Operation *calleeValue) {
  auto callee = CIRGenCallee::forDirect(calleeValue, GlobalDecl(FD));
  return CGF.buildCall(E->getCallee()->getType(), callee, E, ReturnValueSlot());
}

template <class Operation>
static RValue buildUnaryFPBuiltin(CIRGenFunction &CGF, const CallExpr &E) {
  auto Arg = CGF.buildScalarExpr(E.getArg(0));

  CIRGenFunction::CIRGenFPOptionsRAII FPOptsRAII(CGF, &E);
  if (CGF.getBuilder().getIsFPConstrained())
    llvm_unreachable("constraint FP operations are NYI");

  auto Call =
      CGF.getBuilder().create<Operation>(Arg.getLoc(), Arg.getType(), Arg);
  return RValue::get(Call->getResult(0));
}

template <typename Op>
static RValue
buildBuiltinBitOp(CIRGenFunction &CGF, const CallExpr *E,
                  std::optional<CIRGenFunction::BuiltinCheckKind> CK) {
  mlir::Value arg;
  if (CK.has_value())
    arg = CGF.buildCheckedArgForBuiltin(E->getArg(0), *CK);
  else
    arg = CGF.buildScalarExpr(E->getArg(0));

  auto resultTy = CGF.ConvertType(E->getType());
  auto op =
      CGF.getBuilder().create<Op>(CGF.getLoc(E->getExprLoc()), resultTy, arg);
  return RValue::get(op);
}

RValue CIRGenFunction::buildBuiltinExpr(const GlobalDecl GD, unsigned BuiltinID,
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
    if (Result.Val.isFloat())
      llvm_unreachable("NYI");
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

  // There are LLVM math intrinsics/instructions corresponding to math library
  // functions except the LLVM op will never set errno while the math library
  // might. Also, math builtins have the same semantics as their math library
  // twins. Thus, we can transform math library and builtin calls to their
  // LLVM counterparts if the call is marked 'const' (known to never set errno).
  // In case FP exceptions are enabled, the experimental versions of the
  // intrinsics model those.
  bool ConstWithoutErrnoAndExceptions =
      getContext().BuiltinInfo.isConstWithoutErrnoAndExceptions(BuiltinID);
  bool ConstWithoutExceptions =
      getContext().BuiltinInfo.isConstWithoutExceptions(BuiltinID);
  if (FD->hasAttr<ConstAttr>() ||
      ((ConstWithoutErrnoAndExceptions || ConstWithoutExceptions) &&
       (!ConstWithoutErrnoAndExceptions || (!getLangOpts().MathErrno)))) {
    switch (BuiltinIDIfNoAsmLabel) {
    case Builtin::BIceil:
    case Builtin::BIceilf:
    case Builtin::BIceill:
    case Builtin::BI__builtin_ceil:
    case Builtin::BI__builtin_ceilf:
    case Builtin::BI__builtin_ceilf16:
    case Builtin::BI__builtin_ceill:
    case Builtin::BI__builtin_ceilf128:
      return buildUnaryFPBuiltin<mlir::cir::CeilOp>(*this, *E);

    case Builtin::BIcopysign:
    case Builtin::BIcopysignf:
    case Builtin::BIcopysignl:
    case Builtin::BI__builtin_copysign:
    case Builtin::BI__builtin_copysignf:
    case Builtin::BI__builtin_copysignf16:
    case Builtin::BI__builtin_copysignl:
    case Builtin::BI__builtin_copysignf128:
      llvm_unreachable("NYI");

    case Builtin::BIcos:
    case Builtin::BIcosf:
    case Builtin::BIcosl:
    case Builtin::BI__builtin_cos:
    case Builtin::BI__builtin_cosf:
    case Builtin::BI__builtin_cosf16:
    case Builtin::BI__builtin_cosl:
    case Builtin::BI__builtin_cosf128:
      return buildUnaryFPBuiltin<mlir::cir::CosOp>(*this, *E);

    case Builtin::BIexp:
    case Builtin::BIexpf:
    case Builtin::BIexpl:
    case Builtin::BI__builtin_exp:
    case Builtin::BI__builtin_expf:
    case Builtin::BI__builtin_expf16:
    case Builtin::BI__builtin_expl:
    case Builtin::BI__builtin_expf128:
      return buildUnaryFPBuiltin<mlir::cir::ExpOp>(*this, *E);

    case Builtin::BIexp2:
    case Builtin::BIexp2f:
    case Builtin::BIexp2l:
    case Builtin::BI__builtin_exp2:
    case Builtin::BI__builtin_exp2f:
    case Builtin::BI__builtin_exp2f16:
    case Builtin::BI__builtin_exp2l:
    case Builtin::BI__builtin_exp2f128:
      return buildUnaryFPBuiltin<mlir::cir::Exp2Op>(*this, *E);

    case Builtin::BIfabs:
    case Builtin::BIfabsf:
    case Builtin::BIfabsl:
    case Builtin::BI__builtin_fabs:
    case Builtin::BI__builtin_fabsf:
    case Builtin::BI__builtin_fabsf16:
    case Builtin::BI__builtin_fabsl:
    case Builtin::BI__builtin_fabsf128:
      return buildUnaryFPBuiltin<mlir::cir::FAbsOp>(*this, *E);

    case Builtin::BIfloor:
    case Builtin::BIfloorf:
    case Builtin::BIfloorl:
    case Builtin::BI__builtin_floor:
    case Builtin::BI__builtin_floorf:
    case Builtin::BI__builtin_floorf16:
    case Builtin::BI__builtin_floorl:
    case Builtin::BI__builtin_floorf128:
      return buildUnaryFPBuiltin<mlir::cir::FloorOp>(*this, *E);

    case Builtin::BIfma:
    case Builtin::BIfmaf:
    case Builtin::BIfmal:
    case Builtin::BI__builtin_fma:
    case Builtin::BI__builtin_fmaf:
    case Builtin::BI__builtin_fmaf16:
    case Builtin::BI__builtin_fmal:
    case Builtin::BI__builtin_fmaf128:
      llvm_unreachable("NYI");

    case Builtin::BIfmax:
    case Builtin::BIfmaxf:
    case Builtin::BIfmaxl:
    case Builtin::BI__builtin_fmax:
    case Builtin::BI__builtin_fmaxf:
    case Builtin::BI__builtin_fmaxf16:
    case Builtin::BI__builtin_fmaxl:
    case Builtin::BI__builtin_fmaxf128:
      llvm_unreachable("NYI");

    case Builtin::BIfmin:
    case Builtin::BIfminf:
    case Builtin::BIfminl:
    case Builtin::BI__builtin_fmin:
    case Builtin::BI__builtin_fminf:
    case Builtin::BI__builtin_fminf16:
    case Builtin::BI__builtin_fminl:
    case Builtin::BI__builtin_fminf128:
      llvm_unreachable("NYI");

    // fmod() is a special-case. It maps to the frem instruction rather than an
    // LLVM intrinsic.
    case Builtin::BIfmod:
    case Builtin::BIfmodf:
    case Builtin::BIfmodl:
    case Builtin::BI__builtin_fmod:
    case Builtin::BI__builtin_fmodf:
    case Builtin::BI__builtin_fmodf16:
    case Builtin::BI__builtin_fmodl:
    case Builtin::BI__builtin_fmodf128: {
      llvm_unreachable("NYI");
    }

    case Builtin::BIlog:
    case Builtin::BIlogf:
    case Builtin::BIlogl:
    case Builtin::BI__builtin_log:
    case Builtin::BI__builtin_logf:
    case Builtin::BI__builtin_logf16:
    case Builtin::BI__builtin_logl:
    case Builtin::BI__builtin_logf128:
      return buildUnaryFPBuiltin<mlir::cir::LogOp>(*this, *E);

    case Builtin::BIlog10:
    case Builtin::BIlog10f:
    case Builtin::BIlog10l:
    case Builtin::BI__builtin_log10:
    case Builtin::BI__builtin_log10f:
    case Builtin::BI__builtin_log10f16:
    case Builtin::BI__builtin_log10l:
    case Builtin::BI__builtin_log10f128:
      return buildUnaryFPBuiltin<mlir::cir::Log10Op>(*this, *E);

    case Builtin::BIlog2:
    case Builtin::BIlog2f:
    case Builtin::BIlog2l:
    case Builtin::BI__builtin_log2:
    case Builtin::BI__builtin_log2f:
    case Builtin::BI__builtin_log2f16:
    case Builtin::BI__builtin_log2l:
    case Builtin::BI__builtin_log2f128:
      return buildUnaryFPBuiltin<mlir::cir::Log2Op>(*this, *E);

    case Builtin::BInearbyint:
    case Builtin::BInearbyintf:
    case Builtin::BInearbyintl:
    case Builtin::BI__builtin_nearbyint:
    case Builtin::BI__builtin_nearbyintf:
    case Builtin::BI__builtin_nearbyintl:
    case Builtin::BI__builtin_nearbyintf128:
      return buildUnaryFPBuiltin<mlir::cir::NearbyintOp>(*this, *E);

    case Builtin::BIpow:
    case Builtin::BIpowf:
    case Builtin::BIpowl:
    case Builtin::BI__builtin_pow:
    case Builtin::BI__builtin_powf:
    case Builtin::BI__builtin_powf16:
    case Builtin::BI__builtin_powl:
    case Builtin::BI__builtin_powf128:
      llvm_unreachable("NYI");

    case Builtin::BIrint:
    case Builtin::BIrintf:
    case Builtin::BIrintl:
    case Builtin::BI__builtin_rint:
    case Builtin::BI__builtin_rintf:
    case Builtin::BI__builtin_rintf16:
    case Builtin::BI__builtin_rintl:
    case Builtin::BI__builtin_rintf128:
      return buildUnaryFPBuiltin<mlir::cir::RintOp>(*this, *E);

    case Builtin::BIround:
    case Builtin::BIroundf:
    case Builtin::BIroundl:
    case Builtin::BI__builtin_round:
    case Builtin::BI__builtin_roundf:
    case Builtin::BI__builtin_roundf16:
    case Builtin::BI__builtin_roundl:
    case Builtin::BI__builtin_roundf128:
      return buildUnaryFPBuiltin<mlir::cir::RoundOp>(*this, *E);

    case Builtin::BIsin:
    case Builtin::BIsinf:
    case Builtin::BIsinl:
    case Builtin::BI__builtin_sin:
    case Builtin::BI__builtin_sinf:
    case Builtin::BI__builtin_sinf16:
    case Builtin::BI__builtin_sinl:
    case Builtin::BI__builtin_sinf128:
      return buildUnaryFPBuiltin<mlir::cir::SinOp>(*this, *E);

    case Builtin::BIsqrt:
    case Builtin::BIsqrtf:
    case Builtin::BIsqrtl:
    case Builtin::BI__builtin_sqrt:
    case Builtin::BI__builtin_sqrtf:
    case Builtin::BI__builtin_sqrtf16:
    case Builtin::BI__builtin_sqrtl:
    case Builtin::BI__builtin_sqrtf128:
      return buildUnaryFPBuiltin<mlir::cir::SqrtOp>(*this, *E);

    case Builtin::BItrunc:
    case Builtin::BItruncf:
    case Builtin::BItruncl:
    case Builtin::BI__builtin_trunc:
    case Builtin::BI__builtin_truncf:
    case Builtin::BI__builtin_truncf16:
    case Builtin::BI__builtin_truncl:
    case Builtin::BI__builtin_truncf128:
      return buildUnaryFPBuiltin<mlir::cir::TruncOp>(*this, *E);

    case Builtin::BIlround:
    case Builtin::BIlroundf:
    case Builtin::BIlroundl:
    case Builtin::BI__builtin_lround:
    case Builtin::BI__builtin_lroundf:
    case Builtin::BI__builtin_lroundl:
    case Builtin::BI__builtin_lroundf128:
      llvm_unreachable("NYI");

    case Builtin::BIllround:
    case Builtin::BIllroundf:
    case Builtin::BIllroundl:
    case Builtin::BI__builtin_llround:
    case Builtin::BI__builtin_llroundf:
    case Builtin::BI__builtin_llroundl:
    case Builtin::BI__builtin_llroundf128:
      llvm_unreachable("NYI");

    case Builtin::BIlrint:
    case Builtin::BIlrintf:
    case Builtin::BIlrintl:
    case Builtin::BI__builtin_lrint:
    case Builtin::BI__builtin_lrintf:
    case Builtin::BI__builtin_lrintl:
    case Builtin::BI__builtin_lrintf128:
      llvm_unreachable("NYI");

    case Builtin::BIllrint:
    case Builtin::BIllrintf:
    case Builtin::BIllrintl:
    case Builtin::BI__builtin_llrint:
    case Builtin::BI__builtin_llrintf:
    case Builtin::BI__builtin_llrintl:
    case Builtin::BI__builtin_llrintf128:
      llvm_unreachable("NYI");

    default:
      break;
    }
  }

  switch (BuiltinIDIfNoAsmLabel) {
  default:
    break;

  case Builtin::BIprintf:
    if (getTarget().getTriple().isNVPTX() ||
        getTarget().getTriple().isAMDGCN()) {
      llvm_unreachable("NYI");
    }
    break;

  // C stdarg builtins.
  case Builtin::BI__builtin_stdarg_start:
  case Builtin::BI__builtin_va_start:
  case Builtin::BI__va_start:
  case Builtin::BI__builtin_va_end: {
    buildVAStartEnd(BuiltinID == Builtin::BI__va_start
                        ? buildScalarExpr(E->getArg(0))
                        : buildVAListRef(E->getArg(0)).getPointer(),
                    BuiltinID != Builtin::BI__builtin_va_end);
    return {};
  }
  case Builtin::BI__builtin_va_copy: {
    auto dstPtr = buildVAListRef(E->getArg(0)).getPointer();
    auto srcPtr = buildVAListRef(E->getArg(1)).getPointer();
    builder.create<mlir::cir::VACopyOp>(dstPtr.getLoc(), dstPtr, srcPtr);
    return {};
  }

  case Builtin::BI__builtin_expect:
  case Builtin::BI__builtin_expect_with_probability: {
    auto ArgValue = buildScalarExpr(E->getArg(0));
    auto ExpectedValue = buildScalarExpr(E->getArg(1));

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
      ProbAttr = mlir::FloatAttr::get(
          mlir::FloatType::getF64(builder.getContext()), Probability);
    }

    auto result = builder.create<mlir::cir::ExpectOp>(
        getLoc(E->getSourceRange()), ArgValue.getType(), ArgValue,
        ExpectedValue, ProbAttr);

    return RValue::get(result);
  }
  case Builtin::BI__builtin_unpredictable: {
    if (CGM.getCodeGenOpts().OptimizationLevel != 0)
      assert(!UnimplementedFeature::insertBuiltinUnpredictable());
    return RValue::get(buildScalarExpr(E->getArg(0)));
  }

  // C++ std:: builtins.
  case Builtin::BImove:
  case Builtin::BImove_if_noexcept:
  case Builtin::BIforward:
  case Builtin::BIas_const:
    return RValue::get(buildLValue(E->getArg(0)).getPointer());
  case Builtin::BI__GetExceptionInfo: {
    llvm_unreachable("NYI");
  }

  case Builtin::BI__fastfail:
    llvm_unreachable("NYI");

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
    llvm_unreachable("NYI");

  case Builtin::BI__builtin_coro_frame: {
    return buildCoroutineFrame();
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
    fnOp.setBuiltinAttr(mlir::UnitAttr::get(builder.getContext()));
    return buildCall(E->getCallee()->getType(), CIRGenCallee::forDirect(fnOp),
                     E, ReturnValue);
  }
  case Builtin::BI__builtin_dynamic_object_size: {
    // Fallthrough below, assert until we have a testcase.
    llvm_unreachable("NYI");
  }
  case Builtin::BI__builtin_object_size: {
    unsigned Type =
        E->getArg(1)->EvaluateKnownConstInt(getContext()).getZExtValue();
    auto ResType = ConvertType(E->getType()).dyn_cast<mlir::cir::IntType>();
    assert(ResType && "not sure what to do?");

    // We pass this builtin onto the optimizer so that it can figure out the
    // object size in more complex cases.
    bool IsDynamic = BuiltinID == Builtin::BI__builtin_dynamic_object_size;
    return RValue::get(emitBuiltinObjectSize(E->getArg(0), Type, ResType,
                                             /*EmittedE=*/nullptr, IsDynamic));
  }
  case Builtin::BI__builtin_unreachable: {
    buildUnreachable(E->getExprLoc());

    // We do need to preserve an insertion point.
    builder.createBlock(builder.getBlock()->getParent());

    return RValue::get(nullptr);
  }
  case Builtin::BImemcpy:
  case Builtin::BI__builtin_memcpy:
  case Builtin::BImempcpy:
  case Builtin::BI__builtin_mempcpy: {
    Address Dest = buildPointerWithAlignment(E->getArg(0));
    Address Src = buildPointerWithAlignment(E->getArg(1));
    mlir::Value SizeVal = buildScalarExpr(E->getArg(2));
    buildNonNullArgCheck(RValue::get(Dest.getPointer()),
                         E->getArg(0)->getType(), E->getArg(0)->getExprLoc(),
                         FD, 0);
    buildNonNullArgCheck(RValue::get(Src.getPointer()), E->getArg(1)->getType(),
                         E->getArg(1)->getExprLoc(), FD, 1);
    builder.createMemCpy(getLoc(E->getSourceRange()), Dest.getPointer(),
                         Src.getPointer(), SizeVal);
    if (BuiltinID == Builtin::BImempcpy ||
        BuiltinID == Builtin::BI__builtin_mempcpy)
      llvm_unreachable("mempcpy is NYI");
    else
      return RValue::get(Dest.getPointer());
  }

  case Builtin::BI__builtin_clrsb:
  case Builtin::BI__builtin_clrsbl:
  case Builtin::BI__builtin_clrsbll:
    return buildBuiltinBitOp<mlir::cir::BitClrsbOp>(*this, E, std::nullopt);

  case Builtin::BI__builtin_ctzs:
  case Builtin::BI__builtin_ctz:
  case Builtin::BI__builtin_ctzl:
  case Builtin::BI__builtin_ctzll:
    return buildBuiltinBitOp<mlir::cir::BitCtzOp>(*this, E, BCK_CTZPassedZero);

  case Builtin::BI__builtin_clzs:
  case Builtin::BI__builtin_clz:
  case Builtin::BI__builtin_clzl:
  case Builtin::BI__builtin_clzll:
    return buildBuiltinBitOp<mlir::cir::BitClzOp>(*this, E, BCK_CLZPassedZero);

  case Builtin::BI__builtin_ffs:
  case Builtin::BI__builtin_ffsl:
  case Builtin::BI__builtin_ffsll:
    return buildBuiltinBitOp<mlir::cir::BitFfsOp>(*this, E, std::nullopt);

  case Builtin::BI__builtin_parity:
  case Builtin::BI__builtin_parityl:
  case Builtin::BI__builtin_parityll:
    return buildBuiltinBitOp<mlir::cir::BitParityOp>(*this, E, std::nullopt);

  case Builtin::BI__popcnt16:
  case Builtin::BI__popcnt:
  case Builtin::BI__popcnt64:
  case Builtin::BI__builtin_popcount:
  case Builtin::BI__builtin_popcountl:
  case Builtin::BI__builtin_popcountll:
    return buildBuiltinBitOp<mlir::cir::BitPopcountOp>(*this, E, std::nullopt);
  }

  // If this is an alias for a lib function (e.g. __builtin_sin), emit
  // the call using the normal call path, but using the unmangled
  // version of the function name.
  if (getContext().BuiltinInfo.isLibFunction(BuiltinID))
    return buildLibraryCall(*this, FD, E,
                            CGM.getBuiltinLibFunction(FD, BuiltinID));

  // If this is a predefined lib function (e.g. malloc), emit the call
  // using exactly the normal call path.
  if (getContext().BuiltinInfo.isPredefinedLibFunction(BuiltinID))
    return buildLibraryCall(*this, FD, E,
                            buildScalarExpr(E->getCallee()).getDefiningOp());

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
  TypeEvaluationKind EvalKind = getEvaluationKind(E->getType());
  if (EvalKind == TEK_Aggregate && ReturnValue.isNull()) {
    llvm_unreachable("NYI");
  }

  // Now see if we can emit a target-specific builtin.
  if (auto v = buildTargetBuiltinExpr(BuiltinID, E, ReturnValue)) {
    llvm_unreachable("NYI");
  }

  llvm_unreachable("NYI");
  //   ErrorUnsupported(E, "builtin function");

  // Unknown builtin, for now just dump it out and return undef.
  return GetUndefRValue(E->getType());
}

mlir::Value CIRGenFunction::buildCheckedArgForBuiltin(const Expr *E,
                                                      BuiltinCheckKind Kind) {
  assert((Kind == BCK_CLZPassedZero || Kind == BCK_CTZPassedZero) &&
         "Unsupported builtin check kind");

  auto value = buildScalarExpr(E);
  if (!SanOpts.has(SanitizerKind::Builtin))
    return value;

  assert(!UnimplementedFeature::sanitizerBuiltin());
  llvm_unreachable("NYI");
}

static mlir::Value buildTargetArchBuiltinExpr(CIRGenFunction *CGF,
                                              unsigned BuiltinID,
                                              const CallExpr *E,
                                              ReturnValueSlot ReturnValue,
                                              llvm::Triple::ArchType Arch) {
  llvm_unreachable("NYI");
  return {};
}

mlir::Value
CIRGenFunction::buildTargetBuiltinExpr(unsigned BuiltinID, const CallExpr *E,
                                       ReturnValueSlot ReturnValue) {
  if (getContext().BuiltinInfo.isAuxBuiltinID(BuiltinID)) {
    assert(getContext().getAuxTargetInfo() && "Missing aux target info");
    return buildTargetArchBuiltinExpr(
        this, getContext().BuiltinInfo.getAuxBuiltinID(BuiltinID), E,
        ReturnValue, getContext().getAuxTargetInfo()->getTriple().getArch());
  }

  return buildTargetArchBuiltinExpr(this, BuiltinID, E, ReturnValue,
                                    getTarget().getTriple().getArch());
}

void CIRGenFunction::buildVAStartEnd(mlir::Value ArgValue, bool IsStart) {
  // LLVM codegen casts to *i8, no real gain on doing this for CIRGen this
  // early, defer to LLVM lowering.
  if (IsStart)
    builder.create<mlir::cir::VAStartOp>(ArgValue.getLoc(), ArgValue);
  else
    builder.create<mlir::cir::VAEndOp>(ArgValue.getLoc(), ArgValue);
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
                                                  mlir::cir::IntType ResType,
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

      return buildLoadOfScalar(DIter->second, /*Volatile=*/false,
                               getContext().getSizeType(), E->getBeginLoc());
    }
  }

  // LLVM can't handle Type=3 appropriately, and __builtin_object_size shouldn't
  // evaluate E for side-effects. In either case, just like original LLVM
  // lowering, we shouldn't lower to `cir.objsize`.
  if (Type == 3 || (!EmittedE && E->HasSideEffects(getContext())))
    llvm_unreachable("NYI");

  auto Ptr = EmittedE ? EmittedE : buildScalarExpr(E);
  assert(Ptr.getType().isa<mlir::cir::PointerType>() &&
         "Non-pointer passed to __builtin_object_size?");

  // LLVM intrinsics (which CIR lowers to at some point, only supports 0
  // and 2, account for that right now.
  mlir::cir::SizeInfoType sizeInfoTy = ((Type & 2) != 0)
                                           ? mlir::cir::SizeInfoType::min
                                           : mlir::cir::SizeInfoType::max;
  // TODO(cir): Heads up for LLVM lowering, For GCC compatibility,
  // __builtin_object_size treat NULL as unknown size.
  return builder.create<mlir::cir::ObjSizeOp>(
      getLoc(E->getSourceRange()), ResType, Ptr, sizeInfoTy, IsDynamic);
}

mlir::Value CIRGenFunction::evaluateOrEmitBuiltinObjectSize(
    const Expr *E, unsigned Type, mlir::cir::IntType ResType,
    mlir::Value EmittedE, bool IsDynamic) {
  uint64_t ObjectSize;
  if (!E->tryEvaluateObjectSize(ObjectSize, getContext(), Type))
    return emitBuiltinObjectSize(E, Type, ResType, EmittedE, IsDynamic);
  return builder.getConstInt(getLoc(E->getSourceRange()), ResType, ObjectSize);
}

/// Given a builtin id for a function like "__builtin_fabsf", return a Function*
/// for "fabsf".
mlir::cir::FuncOp CIRGenModule::getBuiltinLibFunction(const FunctionDecl *FD,
                                                      unsigned BuiltinID) {
  assert(astCtx.BuiltinInfo.isLibFunction(BuiltinID));

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
      Name = astCtx.BuiltinInfo.getName(BuiltinID).substr(10);
  }

  auto Ty = getTypes().ConvertType(FD->getType());
  return GetOrCreateCIRFunction(Name, Ty, D, /*ForVTable=*/false);
}

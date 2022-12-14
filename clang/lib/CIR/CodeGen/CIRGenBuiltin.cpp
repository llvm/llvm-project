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
#include "llvm/IR/Intrinsics.h"

#include "clang/AST/GlobalDecl.h"
#include "clang/Basic/Builtins.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Value.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"

using namespace cir;
using namespace clang;
using namespace mlir::cir;
using namespace llvm;

RValue CIRGenFunction::buildBuiltinExpr(const GlobalDecl GD, unsigned BuiltinID,
                                        const CallExpr *E,
                                        ReturnValueSlot ReturnValue) {
  const FunctionDecl *FD = GD.getDecl()->getAsFunction();

  // See if we can constant fold this builtin.  If so, don't emit it at all.
  // TODO: Extend this handling to all builtin calls that we can constant-fold.
  Expr::EvalResult Result;
  if (E->isPRValue() && E->EvaluateAsRValue(Result, CGM.getASTContext()) &&
      !Result.hasSideEffects()) {
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
    llvm_unreachable("NYI");
  }

  switch (BuiltinIDIfNoAsmLabel) {
  default:
    llvm_unreachable("NYI");
    break;

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

  case Builtin::BI__builtin_coro_frame:
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
  }

  // If this is an alias for a lib function (e.g. __builtin_sin), emit
  // the call using the normal call path, but using the unmangled
  // version of the function name.
  if (getContext().BuiltinInfo.isLibFunction(BuiltinID))
    llvm_unreachable("NYI");

  // If this is a predefined lib function (e.g. malloc), emit the call
  // using exactly the normal call path.
  if (getContext().BuiltinInfo.isPredefinedLibFunction(BuiltinID))
    llvm_unreachable("NYI");

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

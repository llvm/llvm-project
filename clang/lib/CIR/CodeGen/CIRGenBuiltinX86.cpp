//===---- CIRGenBuiltinX86.cpp - Emit CIR for X86 builtins ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This contains code to emit x86/x86_64 Builtin calls as CIR or a function
// call to be later resolved.
//
//===----------------------------------------------------------------------===//

#include "CIRGenCXXABI.h"
#include "CIRGenCall.h"
#include "CIRGenFunction.h"
#include "CIRGenModule.h"
#include "TargetInfo.h"
#include "clang/CIR/MissingFeatures.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Value.h"
#include "clang/AST/GlobalDecl.h"
#include "clang/Basic/Builtins.h"
#include "clang/Basic/TargetBuiltins.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "llvm/Support/ErrorHandling.h"

using namespace clang;
using namespace clang::CIRGen;
using namespace cir;

static std::optional<CIRGenFunction::MSVCIntrin>
translateX86ToMsvcIntrin(unsigned BuiltinID) {
  using MSVCIntrin = CIRGenFunction::MSVCIntrin;
  switch (BuiltinID) {
  default:
    return std::nullopt;
  case clang::X86::BI_BitScanForward:
  case clang::X86::BI_BitScanForward64:
    return MSVCIntrin::_BitScanForward;
  case clang::X86::BI_BitScanReverse:
  case clang::X86::BI_BitScanReverse64:
    return MSVCIntrin::_BitScanReverse;
  case clang::X86::BI_InterlockedAnd64:
    return MSVCIntrin::_InterlockedAnd;
  case clang::X86::BI_InterlockedCompareExchange128:
    return MSVCIntrin::_InterlockedCompareExchange128;
  case clang::X86::BI_InterlockedExchange64:
    return MSVCIntrin::_InterlockedExchange;
  case clang::X86::BI_InterlockedExchangeAdd64:
    return MSVCIntrin::_InterlockedExchangeAdd;
  case clang::X86::BI_InterlockedExchangeSub64:
    return MSVCIntrin::_InterlockedExchangeSub;
  case clang::X86::BI_InterlockedOr64:
    return MSVCIntrin::_InterlockedOr;
  case clang::X86::BI_InterlockedXor64:
    return MSVCIntrin::_InterlockedXor;
  case clang::X86::BI_InterlockedDecrement64:
    return MSVCIntrin::_InterlockedDecrement;
  case clang::X86::BI_InterlockedIncrement64:
    return MSVCIntrin::_InterlockedIncrement;
  }
  llvm_unreachable("must return from switch");
}

mlir::Value CIRGenFunction::emitX86BuiltinExpr(unsigned BuiltinID,
                                               const CallExpr *E) {
  if (BuiltinID == Builtin::BI__builtin_cpu_is)
    llvm_unreachable("__builtin_cpu_is NYI");
  if (BuiltinID == Builtin::BI__builtin_cpu_supports)
    llvm_unreachable("__builtin_cpu_supports NYI");
  if (BuiltinID == Builtin::BI__builtin_cpu_init)
    llvm_unreachable("__builtin_cpu_init NYI");

  // Handle MSVC intrinsics before argument evaluation to prevent double
  // evaluation.
  if (std::optional<MSVCIntrin> MsvcIntId = translateX86ToMsvcIntrin(BuiltinID))
    llvm_unreachable("translateX86ToMsvcIntrin NYI");

  llvm::SmallVector<mlir::Value, 4> Ops;

  // Find out if any arguments are required to be integer constant expressions.
  unsigned ICEArguments = 0;
  ASTContext::GetBuiltinTypeError Error;
  getContext().GetBuiltinType(BuiltinID, Error, &ICEArguments);
  assert(Error == ASTContext::GE_None && "Should not codegen an error");

  for (unsigned i = 0, e = E->getNumArgs(); i != e; i++) {
    Ops.push_back(emitScalarOrConstFoldImmArg(ICEArguments, i, E));
  }

  switch (BuiltinID) {
  default:
    return nullptr;
  case X86::BI_mm_prefetch: {
    llvm_unreachable("_mm_prefetch NYI");
  }
  case X86::BI_mm_clflush: {
    mlir::Type voidTy = cir::VoidType::get(&getMLIRContext());
    return builder
        .create<cir::LLVMIntrinsicCallOp>(
            getLoc(E->getExprLoc()), builder.getStringAttr("x86.sse2.clflush"),
            voidTy, Ops[0])
        .getResult();
  }
  }
}

//===--- LowerModule.cpp - Lower CIR Module to a Target -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file partially mimics clang/lib/CodeGen/CodeGenModule.cpp. The queries
// are adapted to operate on the CIR dialect, however.
//
//===----------------------------------------------------------------------===//

#include "LowerModule.h"
#include "CIRLowerContext.h"
#include "TargetInfo.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/ErrorHandling.h"

namespace mlir {
namespace cir {

static CIRCXXABI *createCXXABI(LowerModule &CGM) {
  switch (CGM.getCXXABIKind()) {
  case clang::TargetCXXABI::AppleARM64:
  case clang::TargetCXXABI::Fuchsia:
  case clang::TargetCXXABI::GenericAArch64:
  case clang::TargetCXXABI::GenericARM:
  case clang::TargetCXXABI::iOS:
  case clang::TargetCXXABI::WatchOS:
  case clang::TargetCXXABI::GenericMIPS:
  case clang::TargetCXXABI::GenericItanium:
  case clang::TargetCXXABI::WebAssembly:
  case clang::TargetCXXABI::XL:
    return CreateItaniumCXXABI(CGM);
  case clang::TargetCXXABI::Microsoft:
    llvm_unreachable("Windows ABI NYI");
  }

  llvm_unreachable("invalid C++ ABI kind");
}

static std::unique_ptr<TargetLoweringInfo>
createTargetLoweringInfo(LowerModule &LM) {
  const clang::TargetInfo &Target = LM.getTarget();
  const llvm::Triple &Triple = Target.getTriple();

  switch (Triple.getArch()) {
  case llvm::Triple::x86_64: {
    switch (Triple.getOS()) {
    case llvm::Triple::Win32:
      llvm_unreachable("Windows ABI NYI");
    default:
      return createX86_64TargetLoweringInfo(LM, X86AVXABILevel::None);
    }
  }
  default:
    llvm_unreachable("ABI NYI");
  }
}

LowerModule::LowerModule(CIRLowerContext &C, ModuleOp &module, StringAttr DL,
                         const clang::TargetInfo &target,
                         PatternRewriter &rewriter)
    : context(C), module(module), Target(target), ABI(createCXXABI(*this)),
      types(*this, DL.getValue()), rewriter(rewriter) {}

const TargetLoweringInfo &LowerModule::getTargetLoweringInfo() {
  if (!TheTargetCodeGenInfo)
    TheTargetCodeGenInfo = createTargetLoweringInfo(*this);
  return *TheTargetCodeGenInfo;
}

LogicalResult LowerModule::rewriteGlobalFunctionDefinition(FuncOp op,
                                                           LowerModule &state) {
  mlir::OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(op);
  return failure();
}

LogicalResult LowerModule::rewriteFunctionCall(CallOp caller, FuncOp callee) {
  mlir::OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(caller);
  return failure();
}

} // namespace cir
} // namespace mlir

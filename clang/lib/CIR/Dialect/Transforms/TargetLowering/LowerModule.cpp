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
#include "LowerFunction.h"
#include "TargetInfo.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "clang/CIR/Target/AArch64.h"
#include "llvm/Support/ErrorHandling.h"

using MissingFeatures = ::cir::MissingFeatures;
using AArch64ABIKind = ::cir::AArch64ABIKind;
using X86AVXABILevel = ::cir::X86AVXABILevel;

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
  case llvm::Triple::aarch64: {
    AArch64ABIKind Kind = AArch64ABIKind::AAPCS;
    if (Target.getABI() == "darwinpcs")
      llvm_unreachable("DarwinPCS ABI NYI");
    else if (Triple.isOSWindows())
      llvm_unreachable("Windows ABI NYI");
    else if (Target.getABI() == "aapcs-soft")
      llvm_unreachable("AAPCS-soft ABI NYI");

    return createAArch64TargetLoweringInfo(LM, Kind);
  }
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

void LowerModule::setCIRFunctionAttributes(FuncOp GD,
                                           const LowerFunctionInfo &Info,
                                           FuncOp F, bool IsThunk) {
  unsigned CallingConv;
  // NOTE(cir): The method below will update the F function in-place with the
  // proper attributes.
  constructAttributeList(GD.getName(), Info, GD, F, CallingConv,
                         /*AttrOnCallSite=*/false, IsThunk);
  // TODO(cir): Set Function's calling convention.
}

/// Set function attributes for a function declaration.
///
/// This method is based on CodeGenModule::SetFunctionAttributes but it
/// altered to consider only the ABI/Target-related bits.
void LowerModule::setFunctionAttributes(FuncOp oldFn, FuncOp newFn,
                                        bool IsIncompleteFunction,
                                        bool IsThunk) {

  // TODO(cir): There's some special handling from attributes related to LLVM
  // intrinsics. Should we do that here as well?

  // Setup target-specific attributes.
  if (!IsIncompleteFunction)
    setCIRFunctionAttributes(oldFn, getTypes().arrangeGlobalDeclaration(oldFn),
                             newFn, IsThunk);

  // TODO(cir): Handle attributes for returned "this" objects.

  // NOTE(cir): Skipping some linkage and other global value attributes here as
  // it might be better for CIRGen to handle them.

  // TODO(cir): Skipping section attributes here.

  // TODO(cir): Skipping error attributes here.

  // If we plan on emitting this inline builtin, we can't treat it as a builtin.
  if (MissingFeatures::funcDeclIsInlineBuiltinDeclaration()) {
    llvm_unreachable("NYI");
  }

  if (MissingFeatures::funcDeclIsReplaceableGlobalAllocationFunction()) {
    llvm_unreachable("NYI");
  }

  if (MissingFeatures::funcDeclIsCXXConstructorDecl() ||
      MissingFeatures::funcDeclIsCXXDestructorDecl())
    llvm_unreachable("NYI");
  else if (MissingFeatures::funcDeclIsCXXMethodDecl())
    llvm_unreachable("NYI");

  // NOTE(cir) Skipping emissions that depend on codegen options, as well as
  // sanitizers handling here. Do this in CIRGen.

  if (MissingFeatures::langOpts() && MissingFeatures::openMP())
    llvm_unreachable("NYI");

  // NOTE(cir): Skipping more things here that depend on codegen options.

  if (MissingFeatures::extParamInfo()) {
    llvm_unreachable("NYI");
  }
}

/// Rewrites an existing function to conform to the ABI.
///
/// This method is based on CodeGenModule::EmitGlobalFunctionDefinition but it
/// considerably simplified as it tries to remove any CodeGen related code.
LogicalResult LowerModule::rewriteFunctionDefinition(FuncOp op) {
  mlir::OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(op);

  // Get ABI/target-specific function information.
  const LowerFunctionInfo &FI = this->getTypes().arrangeGlobalDeclaration(op);

  // Get ABI/target-specific function type.
  FuncType Ty = this->getTypes().getFunctionType(FI);

  // NOTE(cir): Skipping getAddrOfFunction and getOrCreateCIRFunction methods
  // here, as they are mostly codegen logic.

  // Create a new function with the ABI-specific types.
  FuncOp newFn = cast<FuncOp>(rewriter.cloneWithoutRegions(op));
  newFn.setType(Ty);

  // NOTE(cir): The clone above will preserve any existing attributes. If there
  // are high-level attributes that ought to be dropped, do it here.

  // Set up ABI-specific function attributes.
  setFunctionAttributes(op, newFn, false, /*IsThunk=*/false);
  if (MissingFeatures::extParamInfo()) {
    llvm_unreachable("ExtraAttrs are NYI");
  }

  if (LowerFunction(*this, rewriter, op, newFn)
          .generateCode(op, newFn, FI)
          .failed())
    return failure();

  // Erase original ABI-agnostic function.
  rewriter.eraseOp(op);
  return success();
}

LogicalResult LowerModule::rewriteFunctionCall(CallOp callOp, FuncOp funcOp) {
  mlir::OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(callOp);

  // Create a new function with the ABI-specific calling convention.
  if (LowerFunction(*this, rewriter, funcOp, callOp)
          .rewriteCallOp(callOp)
          .failed())
    return failure();

  return success();
}

} // namespace cir
} // namespace mlir

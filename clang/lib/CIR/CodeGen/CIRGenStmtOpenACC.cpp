//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Emit OpenACC Stmt nodes as CIR code.
//
//===----------------------------------------------------------------------===//

#include "CIRGenBuilder.h"
#include "CIRGenFunction.h"
#include "clang/AST/StmtOpenACC.h"

using namespace clang;
using namespace clang::CIRGen;
using namespace cir;

mlir::LogicalResult
CIRGenFunction::emitOpenACCComputeConstruct(const OpenACCComputeConstruct &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenACC Compute Construct");
  return mlir::failure();
}

mlir::LogicalResult
CIRGenFunction::emitOpenACCLoopConstruct(const OpenACCLoopConstruct &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenACC Loop Construct");
  return mlir::failure();
}
mlir::LogicalResult CIRGenFunction::emitOpenACCCombinedConstruct(
    const OpenACCCombinedConstruct &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenACC Combined Construct");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOpenACCDataConstruct(const OpenACCDataConstruct &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenACC Data Construct");
  return mlir::failure();
}
mlir::LogicalResult CIRGenFunction::emitOpenACCEnterDataConstruct(
    const OpenACCEnterDataConstruct &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenACC EnterData Construct");
  return mlir::failure();
}
mlir::LogicalResult CIRGenFunction::emitOpenACCExitDataConstruct(
    const OpenACCExitDataConstruct &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenACC ExitData Construct");
  return mlir::failure();
}
mlir::LogicalResult CIRGenFunction::emitOpenACCHostDataConstruct(
    const OpenACCHostDataConstruct &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenACC HostData Construct");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOpenACCWaitConstruct(const OpenACCWaitConstruct &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenACC Wait Construct");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOpenACCInitConstruct(const OpenACCInitConstruct &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenACC Init Construct");
  return mlir::failure();
}
mlir::LogicalResult CIRGenFunction::emitOpenACCShutdownConstruct(
    const OpenACCShutdownConstruct &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenACC Shutdown Construct");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOpenACCSetConstruct(const OpenACCSetConstruct &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenACC Set Construct");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOpenACCUpdateConstruct(const OpenACCUpdateConstruct &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenACC Update Construct");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOpenACCAtomicConstruct(const OpenACCAtomicConstruct &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenACC Atomic Construct");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOpenACCCacheConstruct(const OpenACCCacheConstruct &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenACC Cache Construct");
  return mlir::failure();
}

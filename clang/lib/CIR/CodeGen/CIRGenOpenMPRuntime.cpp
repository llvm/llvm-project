//===--- CIRGenStmtOpenMP.cpp - Interface to OpenMP Runtimes --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This provides a class for OpenMP runtime MLIR code generation.
//
//===----------------------------------------------------------------------===//

#include "CIRGenOpenMPRuntime.h"
#include "CIRGenFunction.h"
#include "CIRGenModule.h"

using namespace cir;
using namespace clang;

CIRGenOpenMPRuntime::CIRGenOpenMPRuntime(CIRGenModule &CGM) : CGM(CGM) {}

Address CIRGenOpenMPRuntime::getAddressOfLocalVariable(CIRGenFunction &CGF,
                                                       const VarDecl *VD) {
  assert(!UnimplementedFeature::openMPRuntime());
  return Address::invalid();
}

void CIRGenOpenMPRuntime::checkAndEmitLastprivateConditional(
    CIRGenFunction &CGF, const Expr *LHS) {
  assert(!UnimplementedFeature::openMPRuntime());
  return;
}

void CIRGenOpenMPRuntime::registerTargetGlobalVariable(
    const clang::VarDecl *VD, mlir::cir::GlobalOp globalOp) {
  assert(!UnimplementedFeature::openMPRuntime());
  return;
}

void CIRGenOpenMPRuntime::emitDeferredTargetDecls() const {
  assert(!UnimplementedFeature::openMPRuntime());
  return;
}

void CIRGenOpenMPRuntime::emitFunctionProlog(CIRGenFunction &CGF,
                                             const clang::Decl *D) {
  assert(!UnimplementedFeature::openMPRuntime());
  return;
}

bool CIRGenOpenMPRuntime::emitTargetGlobal(clang::GlobalDecl &GD) {
  assert(!UnimplementedFeature::openMPRuntime());
  return false;
}

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
  assert(!MissingFeatures::openMPRuntime());
  return Address::invalid();
}

void CIRGenOpenMPRuntime::checkAndEmitLastprivateConditional(
    CIRGenFunction &CGF, const Expr *LHS) {
  assert(!MissingFeatures::openMPRuntime());
  return;
}

void CIRGenOpenMPRuntime::registerTargetGlobalVariable(
    const clang::VarDecl *VD, mlir::cir::GlobalOp globalOp) {
  assert(!MissingFeatures::openMPRuntime());
  return;
}

void CIRGenOpenMPRuntime::emitDeferredTargetDecls() const {
  assert(!MissingFeatures::openMPRuntime());
  return;
}

void CIRGenOpenMPRuntime::emitFunctionProlog(CIRGenFunction &CGF,
                                             const clang::Decl *D) {
  assert(!MissingFeatures::openMPRuntime());
  return;
}

bool CIRGenOpenMPRuntime::emitTargetGlobal(clang::GlobalDecl &GD) {
  assert(!MissingFeatures::openMPRuntime());
  return false;
}

void CIRGenOpenMPRuntime::emitTaskWaitCall(CIRGenBuilderTy &builder,
                                           CIRGenFunction &CGF,
                                           mlir::Location Loc,
                                           const OMPTaskDataTy &Data) {

  if (!CGF.HaveInsertPoint())
    return;

  if (CGF.CGM.getLangOpts().OpenMPIRBuilder && Data.Dependences.empty()) {
    // TODO: Need to support taskwait with dependences in the OpenMPIRBuilder.
    // TODO(cir): This could change in the near future when OpenMP 5.0 gets
    // supported by MLIR
    llvm_unreachable("NYI");
    // builder.create<mlir::omp::TaskwaitOp>(Loc);
  } else {
    llvm_unreachable("NYI");
  }
  assert(!MissingFeatures::openMPRegionInfo());
}

void CIRGenOpenMPRuntime::emitBarrierCall(CIRGenBuilderTy &builder,
                                          CIRGenFunction &CGF,
                                          mlir::Location Loc) {

  assert(!MissingFeatures::openMPRegionInfo());

  if (CGF.CGM.getLangOpts().OpenMPIRBuilder) {
    builder.create<mlir::omp::BarrierOp>(Loc);
    return;
  }

  if (!CGF.HaveInsertPoint())
    return;

  llvm_unreachable("NYI");
}

void CIRGenOpenMPRuntime::emitTaskyieldCall(CIRGenBuilderTy &builder,
                                            CIRGenFunction &CGF,
                                            mlir::Location Loc) {

  if (!CGF.HaveInsertPoint())
    return;

  if (CGF.CGM.getLangOpts().OpenMPIRBuilder) {
    builder.create<mlir::omp::TaskyieldOp>(Loc);
  } else {
    llvm_unreachable("NYI");
  }

  assert(!MissingFeatures::openMPRegionInfo());
}

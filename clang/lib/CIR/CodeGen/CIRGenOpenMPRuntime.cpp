//===--- CIRGenOpenMPRuntime.cpp - OpenMP code generation helpers ------=--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Helpers for OpenMP-specific CIR code generation.
//
////===--------------------------------------------------------------------===//

#include "CIRGenOpenMPRuntime.h"
#include "CIRGenModule.h"

#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/OpenMP/OpenMPInterfaces.h"
#include "clang/AST/OpenMPClause.h"

using namespace clang;
using namespace clang::CIRGen;

static mlir::omp::DeclareTargetDeviceType
convertDeviceType(OMPDeclareTargetDeclAttr::DevTypeTy devTy) {
  switch (devTy) {
  case OMPDeclareTargetDeclAttr::DT_Host:
    return mlir::omp::DeclareTargetDeviceType::host;
  case OMPDeclareTargetDeclAttr::DT_NoHost:
    return mlir::omp::DeclareTargetDeviceType::nohost;
  case OMPDeclareTargetDeclAttr::DT_Any:
    return mlir::omp::DeclareTargetDeviceType::any;
  }
  llvm_unreachable("unexpected device type");
}

static mlir::omp::DeclareTargetCaptureClause
convertCaptureClause(OMPDeclareTargetDeclAttr::MapTypeTy mapTy) {
  switch (mapTy) {
  case OMPDeclareTargetDeclAttr::MT_To:
    return mlir::omp::DeclareTargetCaptureClause::to;
  case OMPDeclareTargetDeclAttr::MT_Enter:
    return mlir::omp::DeclareTargetCaptureClause::enter;
  case OMPDeclareTargetDeclAttr::MT_Link:
    return mlir::omp::DeclareTargetCaptureClause::link;
  case OMPDeclareTargetDeclAttr::MT_Local:
    return mlir::omp::DeclareTargetCaptureClause::none;
  }
  llvm_unreachable("unexpected map type");
}

/// Returns true if the declaration should be skipped based on its
/// device_type attribute and the current compilation mode.
static bool isAssumedToBeNotEmitted(const ValueDecl *vd, bool isDevice) {
  std::optional<OMPDeclareTargetDeclAttr::DevTypeTy> devTy =
      OMPDeclareTargetDeclAttr::getDeviceType(vd);
  if (!devTy)
    return false;
  // Do not emit device_type(nohost) functions for the host.
  if (!isDevice && *devTy == OMPDeclareTargetDeclAttr::DT_NoHost)
    return true;
  // Do not emit device_type(host) functions for the device.
  if (isDevice && *devTy == OMPDeclareTargetDeclAttr::DT_Host)
    return true;
  return false;
}

/// Recursively check whether the statement tree contains any OpenMP target
/// execution directive (e.g. 'omp target', 'omp target parallel', etc.).
/// Used to identify host functions that must be emitted on the device because
/// they contain target regions that will be outlined during MLIR lowering.
static bool containsTargetRegion(const Stmt *s) {
  if (!s)
    return false;
  if (const auto *e = dyn_cast<OMPExecutableDirective>(s))
    if (isOpenMPTargetExecutionDirective(e->getDirectiveKind()))
      return true;
  for (const Stmt *child : s->children())
    if (containsTargetRegion(child))
      return true;
  return false;
}

bool CIRGenOpenMPRuntime::emitTargetFunctions(GlobalDecl gd) {
  bool isDevice = cgm.getLangOpts().OpenMPIsTargetDevice;

  if (!isDevice) {
    if (const auto *fd = dyn_cast<FunctionDecl>(gd.getDecl()))
      if (isAssumedToBeNotEmitted(cast<ValueDecl>(fd), isDevice))
        return true;
    return false;
  }

  const auto *vd = cast<ValueDecl>(gd.getDecl());

  if (const auto *fd = dyn_cast<FunctionDecl>(vd))
    if (isAssumedToBeNotEmitted(cast<ValueDecl>(fd), isDevice))
      return true;

  // Do not emit function if it is not marked as declare target.
  if (OMPDeclareTargetDeclAttr::isDeclareTargetDeclaration(vd) ||
      alreadyEmittedTargetDecls.count(vd) != 0)
    return false;

  // We must also host functions that contain target regions,
  // because the omp.target ops are nested inside the host function rather than
  // being outlined early. The containsTargetRegion check handles this.
  if (const auto *fd = dyn_cast<FunctionDecl>(vd))
    if (fd->doesThisDeclarationHaveABody() &&
        containsTargetRegion(fd->getBody()))
      return false;

  return true;
}

bool CIRGenOpenMPRuntime::emitTargetGlobalVariable(GlobalDecl gd) {
  if (isAssumedToBeNotEmitted(cast<ValueDecl>(gd.getDecl()),
                              cgm.getLangOpts().OpenMPIsTargetDevice))
    return true;

  if (!cgm.getLangOpts().OpenMPIsTargetDevice)
    return false;

  // We do not need to scan for target regions since there's not early
  // outlining like in OGCG, they will be emitted as omp.target ops instead.

  const auto *vd = cast<VarDecl>(gd.getDecl());

  // Do not emit variable if it is not marked as declare target.
  // OGCG also defers link-clause and USM variables here; we emit errorNYI
  // for those since they are not yet supported.
  std::optional<OMPDeclareTargetDeclAttr::MapTypeTy> res =
      OMPDeclareTargetDeclAttr::isDeclareTargetDeclaration(vd);
  if (!res || *res == OMPDeclareTargetDeclAttr::MT_Link ||
      ((*res == OMPDeclareTargetDeclAttr::MT_To ||
        *res == OMPDeclareTargetDeclAttr::MT_Enter) &&
       false /* NYI: HasRequiresUnifiedSharedMemory */)) {
    if (res && *res == OMPDeclareTargetDeclAttr::MT_Link)
      cgm.errorNYI(vd->getSourceRange(),
                   "declare target global variable with link clause");
    // OGCG defers these variables for later emission. We skip them for now.
    return true;
  }
  return false;
}

// Mirrors CGOpenMPRuntime::emitTargetGlobal.
bool CIRGenOpenMPRuntime::emitTargetGlobal(GlobalDecl gd) {
  if (isa<FunctionDecl>(gd.getDecl()) ||
      isa<OMPDeclareReductionDecl>(gd.getDecl()))
    return emitTargetFunctions(gd);

  return emitTargetGlobalVariable(gd);
}

bool CIRGenOpenMPRuntime::markAsGlobalTarget(GlobalDecl gd) {
  if (!cgm.getLangOpts().OpenMPIsTargetDevice)
    return true;

  const auto *d = cast<FunctionDecl>(gd.getDecl());

  if (OMPDeclareTargetDeclAttr::isDeclareTargetDeclaration(d)) {
    if (d->hasBody() && alreadyEmittedTargetDecls.count(d) == 0) {
      auto f = dyn_cast_if_present<cir::FuncOp>(
          cgm.getGlobalValue(cgm.getMangledName(gd)));
      if (f)
        return !f.isDeclaration();
      return false;
    }
    return true;
  }

  return !alreadyEmittedTargetDecls.insert(d).second;
}

void CIRGenOpenMPRuntime::emitDeclareTargetFunction(const FunctionDecl *fd,
                                                    cir::FuncOp funcOp) {
  const auto *attr = fd->getAttr<OMPDeclareTargetDeclAttr>();
  assert(attr && "expected OMPDeclareTargetDeclAttr");

  // Handles the 'indirect' clause here by creating a global variable to hold
  // the device function address for runtime resolution of indirect calls on
  // the device.
  if (std::optional<OMPDeclareTargetDeclAttr *> activeAttr =
          OMPDeclareTargetDeclAttr::getActiveAttr(fd))
    if ((*activeAttr)->getIndirect())
      cgm.errorNYI(fd->getSourceRange(),
                   "declare target function with indirect clause");

  auto declTargetIface =
      llvm::cast<mlir::omp::DeclareTargetInterface>(funcOp.getOperation());
  declTargetIface.setDeclareTarget(convertDeviceType(attr->getDevType()),
                                   convertCaptureClause(attr->getMapType()),
                                   /*automap=*/false);
}

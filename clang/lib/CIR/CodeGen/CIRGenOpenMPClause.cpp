//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// OpenMP clause emitter implementation. See CIRGenOpenMPClause.h.
//
//===----------------------------------------------------------------------===//

#include "CIRGenOpenMPClause.h"
#include "CIRGenFunction.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "clang/Basic/OpenMPKinds.h"

using namespace clang;
using namespace clang::CIRGen;

static mlir::omp::ClauseMapFlags
mapClauseKindToFlags(OpenMPMapClauseKind kind) {
  switch (kind) {
  case OMPC_MAP_to:
    return mlir::omp::ClauseMapFlags::to;
  case OMPC_MAP_from:
    return mlir::omp::ClauseMapFlags::from;
  case OMPC_MAP_tofrom:
    return mlir::omp::ClauseMapFlags::to | mlir::omp::ClauseMapFlags::from;
  case OMPC_MAP_alloc:
  case OMPC_MAP_release:
    return mlir::omp::ClauseMapFlags::storage;
  case OMPC_MAP_delete:
    return mlir::omp::ClauseMapFlags::del;
  default:
    return mlir::omp::ClauseMapFlags::none;
  }
}

static mlir::Value emitMapInfoForVar(CIRGenFunction &cgf,
                                     mlir::OpBuilder &builder,
                                     mlir::Location loc, const VarDecl *vd,
                                     mlir::omp::ClauseMapFlags mapFlags) {
  Address addr = cgf.getAddrOfLocalVar(vd);
  mlir::Value varPtr = addr.getPointer();
  auto varPtrType = mlir::cast<cir::PointerType>(varPtr.getType());
  mlir::Type elementType = varPtrType.getPointee();

  // Cast to generic pointer if needed.
  if (varPtrType.getAddrSpace()) {
    auto genericPtrType =
        cir::PointerType::get(builder.getContext(), elementType);
    varPtr = cir::CastOp::create(builder, loc, genericPtrType,
                                 cir::CastKind::address_space, varPtr);
    varPtrType = genericPtrType;
  }

  return mlir::omp::MapInfoOp::create(
      builder, loc,
      /*omp_ptr=*/varPtrType,
      /*var_ptr=*/varPtr,
      /*var_ptr_type=*/mlir::TypeAttr::get(elementType),
      /*map_type=*/builder.getAttr<mlir::omp::ClauseMapFlagsAttr>(mapFlags),
      /*map_capture_type=*/
      builder.getAttr<mlir::omp::VariableCaptureKindAttr>(
          mlir::omp::VariableCaptureKind::ByRef),
      /*var_ptr_ptr=*/mlir::Value{},
      /*var_ptr_ptr_type=*/mlir::TypeAttr{},
      /*members=*/mlir::ValueRange{},
      /*members_index=*/mlir::ArrayAttr{},
      /*bounds=*/mlir::ValueRange{},
      /*mapper_id=*/mlir::FlatSymbolRefAttr{},
      /*name=*/builder.getStringAttr(vd->getName()),
      /*partial_map=*/builder.getBoolAttr(false));
}

bool OpenMPClauseEmitter::emitProcBind(
    mlir::omp::ProcBindClauseOps &result) const {
  for (const OMPClause *clause : clauses) {
    const auto *pbc = dyn_cast<OMPProcBindClause>(clause);
    if (!pbc)
      continue;

    switch (pbc->getProcBindKind()) {
    case llvm::omp::ProcBindKind::OMP_PROC_BIND_master:
      result.procBindKind = mlir::omp::ClauseProcBindKindAttr::get(
          builder.getContext(), mlir::omp::ClauseProcBindKind::Master);
      break;
    case llvm::omp::ProcBindKind::OMP_PROC_BIND_close:
      result.procBindKind = mlir::omp::ClauseProcBindKindAttr::get(
          builder.getContext(), mlir::omp::ClauseProcBindKind::Close);
      break;
    case llvm::omp::ProcBindKind::OMP_PROC_BIND_spread:
      result.procBindKind = mlir::omp::ClauseProcBindKindAttr::get(
          builder.getContext(), mlir::omp::ClauseProcBindKind::Spread);
      break;
    case llvm::omp::ProcBindKind::OMP_PROC_BIND_primary:
      result.procBindKind = mlir::omp::ClauseProcBindKindAttr::get(
          builder.getContext(), mlir::omp::ClauseProcBindKind::Primary);
      break;
    case llvm::omp::ProcBindKind::OMP_PROC_BIND_default:
      break;
    case llvm::omp::ProcBindKind::OMP_PROC_BIND_unknown:
      llvm_unreachable("unknown proc-bind kind");
    }
    return true;
  }
  return false;
}

bool OpenMPClauseEmitter::emitMap(
    mlir::omp::MapClauseOps &result,
    llvm::SmallVectorImpl<const VarDecl *> *mapSyms) const {
  bool found = false;
  for (const OMPClause *clause : clauses) {
    const auto *mc = dyn_cast<OMPMapClause>(clause);
    if (!mc)
      continue;

    found = true;

    for (OpenMPMapModifierKind mod : mc->getMapTypeModifiers()) {
      if (mod != OMPC_MAP_MODIFIER_unknown)
        cgm.errorNYI(mc->getBeginLoc(),
                     std::string("OpenMP map modifier '") +
                         getOpenMPSimpleClauseTypeName(
                             llvm::omp::Clause::OMPC_map, mod) +
                         "'");
    }

    if (mc->isImplicit()) {
      cgm.errorNYI(mc->getBeginLoc(), "OpenMP implicit map clause");
      continue;
    }

    mlir::omp::ClauseMapFlags mapFlags = mapClauseKindToFlags(mc->getMapType());

    for (const Expr *varExpr : mc->varlist()) {
      const auto *refExpr = dyn_cast<DeclRefExpr>(varExpr->IgnoreImplicit());
      if (!refExpr) {
        cgm.errorNYI(varExpr->getExprLoc(),
                     "OpenMP map clause with non-DeclRefExpr variable");
        continue;
      }

      const auto *vd = dyn_cast<VarDecl>(refExpr->getDecl());
      if (!vd) {
        cgm.errorNYI(varExpr->getExprLoc(),
                     "OpenMP map clause with non-VarDecl variable");
        continue;
      }

      result.mapVars.push_back(
          emitMapInfoForVar(cgf, builder, loc, vd, mapFlags));
      if (mapSyms)
        mapSyms->push_back(vd);
    }
  }
  return found;
}

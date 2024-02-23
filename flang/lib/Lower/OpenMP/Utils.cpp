//===-- Utils..cpp ----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Coding style: https://mlir.llvm.org/getting_started/DeveloperGuide/
//
//===----------------------------------------------------------------------===//

#include "Utils.h"

#include <flang/Lower/AbstractConverter.h>
#include <flang/Lower/ConvertType.h>
#include <flang/Optimizer/Builder/FIRBuilder.h>
#include <flang/Parser/parse-tree.h>
#include <flang/Parser/tools.h>
#include <flang/Semantics/tools.h>
#include <llvm/Support/CommandLine.h>

llvm::cl::opt<bool> treatIndexAsSection(
    "openmp-treat-index-as-section",
    llvm::cl::desc("In the OpenMP data clauses treat `a(N)` as `a(N:N)`."),
    llvm::cl::init(true));

namespace Fortran {
namespace lower {
namespace omp {

void genObjectList(const Fortran::parser::OmpObjectList &objectList,
                   Fortran::lower::AbstractConverter &converter,
                   llvm::SmallVectorImpl<mlir::Value> &operands) {
  auto addOperands = [&](Fortran::lower::SymbolRef sym) {
    const mlir::Value variable = converter.getSymbolAddress(sym);
    if (variable) {
      operands.push_back(variable);
    } else {
      if (const auto *details =
              sym->detailsIf<Fortran::semantics::HostAssocDetails>()) {
        operands.push_back(converter.getSymbolAddress(details->symbol()));
        converter.copySymbolBinding(details->symbol(), sym);
      }
    }
  };
  for (const Fortran::parser::OmpObject &ompObject : objectList.v) {
    Fortran::semantics::Symbol *sym = getOmpObjectSymbol(ompObject);
    addOperands(*sym);
  }
}

void gatherFuncAndVarSyms(
    const Fortran::parser::OmpObjectList &objList,
    mlir::omp::DeclareTargetCaptureClause clause,
    llvm::SmallVectorImpl<DeclareTargetCapturePair> &symbolAndClause) {
  for (const Fortran::parser::OmpObject &ompObject : objList.v) {
    Fortran::common::visit(
        Fortran::common::visitors{
            [&](const Fortran::parser::Designator &designator) {
              if (const Fortran::parser::Name *name =
                      Fortran::semantics::getDesignatorNameIfDataRef(
                          designator)) {
                symbolAndClause.emplace_back(clause, *name->symbol);
              }
            },
            [&](const Fortran::parser::Name &name) {
              symbolAndClause.emplace_back(clause, *name.symbol);
            }},
        ompObject.u);
  }
}

void checkAndApplyDeclTargetMapFlags(
    Fortran::lower::AbstractConverter &converter,
    llvm::omp::OpenMPOffloadMappingFlags &mapFlags,
    const Fortran::semantics::Symbol &symbol) {
  if (auto declareTargetOp =
          llvm::dyn_cast_if_present<mlir::omp::DeclareTargetInterface>(
              converter.getModuleOp().lookupSymbol(
                  converter.mangleName(symbol)))) {
    // Only Link clauses have OMP_MAP_PTR_AND_OBJ applied, To clause
    // seems to function differently.
    if (declareTargetOp.getDeclareTargetCaptureClause() ==
        mlir::omp::DeclareTargetCaptureClause::link)
      mapFlags |= llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_PTR_AND_OBJ;
  }
}

int findComponentMemberPlacement(
    const Fortran::semantics::Symbol *dTypeSym,
    const Fortran::semantics::Symbol *componentSym) {
  const auto *derived =
      dTypeSym->detailsIf<Fortran::semantics::DerivedTypeDetails>();
  if (!derived)
    return -1;

  int placement = 0;
  for (auto t : derived->componentNames()) {
    if (t == componentSym->name())
      return placement;
    placement++;
  }

  return -1;
}

void insertChildMapInfoIntoParent(
    Fortran::lower::AbstractConverter &converter,
    llvm::SmallVector<const Fortran::semantics::Symbol *> &memberParentSyms,
    llvm::SmallVector<mlir::omp::MapInfoOp> &memberMaps,
    llvm::SmallVector<mlir::Attribute> &memberPlacementIndices,
    llvm::SmallVectorImpl<mlir::Value> &mapOperands,
    llvm::SmallVectorImpl<mlir::Type> *mapSymTypes,
    llvm::SmallVectorImpl<mlir::Location> *mapSymLocs,
    llvm::SmallVectorImpl<const Fortran::semantics::Symbol *> *mapSymbols) {
  // TODO: For multi-nested record types the top level parent is currently
  // the containing parent for all member operations.
  for (auto [idx, sym] : llvm::enumerate(memberParentSyms)) {
    bool parentExists = false;
    size_t parentIdx = 0;
    for (size_t i = 0; i < mapSymbols->size(); ++i) {
      if ((*mapSymbols)[i] == sym) {
        parentExists = true;
        parentIdx = i;
        break;
      }
    }

    if (parentExists) {
      auto mapOp = mlir::dyn_cast<mlir::omp::MapInfoOp>(
          mapOperands[parentIdx].getDefiningOp());
      assert(mapOp && "Parent provided to insertChildMapInfoIntoParent was not "
                      "an expected MapInfoOp");

      // found a parent, append.
      mapOp.getMembersMutable().append((mlir::Value)memberMaps[idx]);
      llvm::SmallVector<mlir::Attribute> memberIndexTmp{
          mapOp.getMembersIndexAttr().begin(),
          mapOp.getMembersIndexAttr().end()};
      memberIndexTmp.push_back(memberPlacementIndices[idx]);
      mapOp.setMembersIndexAttr(mlir::ArrayAttr::get(
          converter.getFirOpBuilder().getContext(), memberIndexTmp));
    } else {
      // NOTE: We take the map type of the first child, this may not
      // be the correct thing to do, however, we shall see. For the moment
      // it allows this to work with enter and exit without causing MLIR
      // verification issues. The more appropriate thing may be to take
      // the "main" map type clause from the directive being used.
      uint64_t mapType = memberMaps[idx].getMapType().value_or(0);

      // create parent to emplace and bind members
      auto origSymbol = converter.getSymbolAddress(*sym);
      mlir::Value mapOp = createMapInfoOp(
          converter.getFirOpBuilder(), origSymbol.getLoc(), origSymbol,
          mlir::Value(), sym->name().ToString(), {}, {memberMaps[idx]},
          mlir::ArrayAttr::get(converter.getFirOpBuilder().getContext(),
                               memberPlacementIndices[idx]),
          mapType, mlir::omp::VariableCaptureKind::ByRef, origSymbol.getType(),
          true);

      mapOperands.push_back(mapOp);
      if (mapSymTypes)
        mapSymTypes->push_back(mapOp.getType());
      if (mapSymLocs)
        mapSymLocs->push_back(mapOp.getLoc());
      if (mapSymbols)
        mapSymbols->push_back(sym);
    }
  }
}

Fortran::semantics::Symbol *
getOmpObjectSymbol(const Fortran::parser::OmpObject &ompObject) {
  Fortran::semantics::Symbol *sym = nullptr;
  std::visit(
      Fortran::common::visitors{
          [&](const Fortran::parser::Designator &designator) {
            if (auto *arrayEle =
                    Fortran::parser::Unwrap<Fortran::parser::ArrayElement>(
                        designator)) {
              sym = GetLastName(arrayEle->base).symbol;
            } else if (auto *structComp = Fortran::parser::Unwrap<
                           Fortran::parser::StructureComponent>(designator)) {
              sym = structComp->component.symbol;
            } else if (const Fortran::parser::Name *name =
                           Fortran::semantics::getDesignatorNameIfDataRef(
                               designator)) {
              sym = name->symbol;
            }
          },
          [&](const Fortran::parser::Name &name) { sym = name.symbol; }},
      ompObject.u);
  return sym;
}

} // namespace omp
} // namespace lower
} // namespace Fortran

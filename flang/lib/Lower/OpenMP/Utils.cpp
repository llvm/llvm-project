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
#include "Clauses.h"

#include <flang/Lower/AbstractConverter.h>
#include <flang/Lower/ConvertType.h>
#include <flang/Optimizer/Builder/FIRBuilder.h>
#include <flang/Parser/parse-tree.h>
#include <flang/Parser/tools.h>
#include <flang/Semantics/tools.h>
#include <llvm/Support/CommandLine.h>

#include <algorithm>
#include <numeric>

llvm::cl::opt<bool> treatIndexAsSection(
    "openmp-treat-index-as-section",
    llvm::cl::desc("In the OpenMP data clauses treat `a(N)` as `a(N:N)`."),
    llvm::cl::init(true));

llvm::cl::opt<bool> enableDelayedPrivatization(
    "openmp-enable-delayed-privatization",
    llvm::cl::desc(
        "Emit `[first]private` variables as clauses on the MLIR ops."),
    llvm::cl::init(false));

namespace Fortran {
namespace lower {
namespace omp {

void genObjectList(const ObjectList &objects,
                   Fortran::lower::AbstractConverter &converter,
                   llvm::SmallVectorImpl<mlir::Value> &operands) {
  for (const Object &object : objects) {
    const Fortran::semantics::Symbol *sym = object.id();
    assert(sym && "Expected Symbol");
    if (mlir::Value variable = converter.getSymbolAddress(*sym)) {
      operands.push_back(variable);
    } else if (const auto *details =
                   sym->detailsIf<Fortran::semantics::HostAssocDetails>()) {
      operands.push_back(converter.getSymbolAddress(details->symbol()));
      converter.copySymbolBinding(details->symbol(), *sym);
    }
  }
}

void genObjectList2(const Fortran::parser::OmpObjectList &objectList,
                    Fortran::lower::AbstractConverter &converter,
                    llvm::SmallVectorImpl<mlir::Value> &operands) {
  auto addOperands = [&](Fortran::lower::SymbolRef sym) {
    const mlir::Value variable = converter.getSymbolAddress(sym);
    if (variable) {
      operands.push_back(variable);
    } else if (const auto *details =
                   sym->detailsIf<Fortran::semantics::HostAssocDetails>()) {
      operands.push_back(converter.getSymbolAddress(details->symbol()));
      converter.copySymbolBinding(details->symbol(), sym);
    }
  };
  for (const Fortran::parser::OmpObject &ompObject : objectList.v) {
    Fortran::semantics::Symbol *sym = getOmpObjectSymbol(ompObject);
    addOperands(*sym);
  }
}

mlir::Type getLoopVarType(Fortran::lower::AbstractConverter &converter,
                          std::size_t loopVarTypeSize) {
  // OpenMP runtime requires 32-bit or 64-bit loop variables.
  loopVarTypeSize = loopVarTypeSize * 8;
  if (loopVarTypeSize < 32) {
    loopVarTypeSize = 32;
  } else if (loopVarTypeSize > 64) {
    loopVarTypeSize = 64;
    mlir::emitWarning(converter.getCurrentLocation(),
                      "OpenMP loop iteration variable cannot have more than 64 "
                      "bits size and will be narrowed into 64 bits.");
  }
  assert((loopVarTypeSize == 32 || loopVarTypeSize == 64) &&
         "OpenMP loop iteration variable size must be transformed into 32-bit "
         "or 64-bit");
  return converter.getFirOpBuilder().getIntegerType(loopVarTypeSize);
}

void gatherFuncAndVarSyms(
    const ObjectList &objects, mlir::omp::DeclareTargetCaptureClause clause,
    llvm::SmallVectorImpl<DeclareTargetCapturePair> &symbolAndClause) {
  for (const Object &object : objects)
    symbolAndClause.emplace_back(clause, *object.id());
}

int getComponentPlacementInParent(
    const Fortran::semantics::Symbol *componentSym) {
  const auto *derived =
      componentSym->owner()
          .derivedTypeSpec()
          ->typeSymbol()
          .detailsIf<Fortran::semantics::DerivedTypeDetails>();
  assert(derived &&
         "expected derived type details when processing component symbol");
  int placement = 0;
  for (auto t : derived->componentNames()) {
    if (t == componentSym->name())
      return placement;
    placement++;
  }
  return -1;
}

std::optional<Object>
getCompObjOrNull(std::optional<Object> object,
                 Fortran::semantics::SemanticsContext &semaCtx) {
  if (!object)
    return std::nullopt;

  auto ref = evaluate::ExtractDataRef(*object.value().ref());
  if (!ref)
    return std::nullopt;

  if (std::get_if<evaluate::Component>(&ref->u))
    return object;

  auto baseObj = getBaseObject(object.value(), semaCtx);
  if (!baseObj)
    return std::nullopt;

  return getCompObjOrNull(baseObj.value(), semaCtx);
}

llvm::SmallVector<int>
generateMemberPlacementIndices(const Object &object,
                               Fortran::semantics::SemanticsContext &semaCtx) {
  std::list<int> indices;
  auto compObj = getCompObjOrNull(object, semaCtx);
  while (compObj) {
    indices.push_front(getComponentPlacementInParent(compObj->id()));
    compObj =
        getCompObjOrNull(getBaseObject(compObj.value(), semaCtx), semaCtx);
  }

  return llvm::SmallVector<int>{std::begin(indices), std::end(indices)};
}

static void calculateShapeAndFillIndices(
    llvm::SmallVectorImpl<int64_t> &shape,
    llvm::SmallVector<OmpMapMemberIndicesData> &memberPlacementData) {
  shape.push_back(memberPlacementData.size());
  size_t largestIndicesSize =
      std::max_element(memberPlacementData.begin(), memberPlacementData.end(),
                       [](auto a, auto b) {
                         return a.memberPlacementIndices.size() <
                                b.memberPlacementIndices.size();
                       })
          ->memberPlacementIndices.size();
  shape.push_back(largestIndicesSize);

  // DenseElementsAttr expects a rectangular shape for the data, so all
  // index lists have to be of the same length, this implaces -1 as filler
  // values
  for (auto &v : memberPlacementData)
    if (v.memberPlacementIndices.size() < largestIndicesSize) {
      auto *prevEnd = v.memberPlacementIndices.end();
      v.memberPlacementIndices.resize(largestIndicesSize);
      std::fill(prevEnd, v.memberPlacementIndices.end(), -1);
    }
}

mlir::DenseIntElementsAttr createDenseElementsAttrFromIndices(
    llvm::SmallVector<OmpMapMemberIndicesData> &memberPlacementData,
    fir::FirOpBuilder &builder) {
  llvm::SmallVector<int64_t> shape;
  calculateShapeAndFillIndices(shape, memberPlacementData);

  llvm::SmallVector<int> indicesFlattened = std::accumulate(
      memberPlacementData.begin(), memberPlacementData.end(),
      llvm::SmallVector<int>(),
      [](llvm::SmallVector<int> &x, OmpMapMemberIndicesData &y) {
        x.insert(x.end(), y.memberPlacementIndices.begin(),
                 y.memberPlacementIndices.end());
        return x;
      });

  return mlir::DenseIntElementsAttr::get(
      mlir::VectorType::get(llvm::ArrayRef<int64_t>(shape),
                            mlir::IntegerType::get(builder.getContext(), 32)),
      llvm::ArrayRef<int32_t>(indicesFlattened));
}

void insertChildMapInfoIntoParent(
    Fortran::lower::AbstractConverter &converter,
    std::map<const Fortran::semantics::Symbol *,
             llvm::SmallVector<OmpMapMemberIndicesData>> &parentMemberIndices,
    llvm::SmallVectorImpl<mlir::Value> &mapOperands,
    llvm::SmallVectorImpl<mlir::Type> *mapSymTypes,
    llvm::SmallVectorImpl<mlir::Location> *mapSymLocs,
    llvm::SmallVectorImpl<const Fortran::semantics::Symbol *> *mapSymbols) {
  for (auto indices : parentMemberIndices) {
    bool parentExists = false;
    size_t parentIdx;
    for (parentIdx = 0; parentIdx < mapSymbols->size(); ++parentIdx)
      if ((*mapSymbols)[parentIdx] == indices.first)
        parentExists = true;

    if (parentExists) {
      auto mapOp = mlir::dyn_cast<mlir::omp::MapInfoOp>(
          mapOperands[parentIdx].getDefiningOp());
      assert(mapOp && "Parent provided to insertChildMapInfoIntoParent was not "
                      "an expected MapInfoOp");

      for (auto memberIndicesData : indices.second)
        mapOp.getMembersMutable().append(
            (mlir::Value)memberIndicesData.memberMap);

      mapOp.setMembersIndexAttr(createDenseElementsAttrFromIndices(
          indices.second, converter.getFirOpBuilder()));
    } else {
      // NOTE: We take the map type of the first child, this may not
      // be the correct thing to do, however, we shall see. For the moment
      // it allows this to work with enter and exit without causing MLIR
      // verification issues. The more appropriate thing may be to take
      // the "main" map type clause from the directive being used.
      uint64_t mapType = indices.second[0].memberMap.getMapType().value_or(0);

      // create parent to emplace and bind members
      auto origSymbol = converter.getSymbolAddress(*indices.first);

      llvm::SmallVector<mlir::Value> members;
      for (auto memberIndicesData : indices.second)
        members.push_back((mlir::Value)memberIndicesData.memberMap);

      mlir::Value mapOp = createMapInfoOp(
          converter.getFirOpBuilder(), origSymbol.getLoc(), origSymbol,
          mlir::Value(), indices.first->name().ToString(), {}, members,
          createDenseElementsAttrFromIndices(indices.second,
                                             converter.getFirOpBuilder()),
          mapType, mlir::omp::VariableCaptureKind::ByRef, origSymbol.getType(),
          true);

      mapOperands.push_back(mapOp);
      if (mapSymTypes)
        mapSymTypes->push_back(mapOp.getType());
      if (mapSymLocs)
        mapSymLocs->push_back(mapOp.getLoc());
      if (mapSymbols)
        mapSymbols->push_back(indices.first);
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

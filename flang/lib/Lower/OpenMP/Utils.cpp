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
#include <flang/Lower/PFTBuilder.h>
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

int64_t getCollapseValue(const List<Clause> &clauses) {
  auto iter = llvm::find_if(clauses, [](const Clause &clause) {
    return clause.id == llvm::omp::Clause::OMPC_collapse;
  });
  if (iter != clauses.end()) {
    const auto &collapse = std::get<clause::Collapse>(iter->u);
    return evaluate::ToInt64(collapse.v).value();
  }
  return 1;
}

uint32_t getOpenMPVersion(mlir::ModuleOp mod) {
  if (mlir::Attribute verAttr = mod->getAttr("omp.version"))
    return llvm::cast<mlir::omp::VersionAttr>(verAttr).getVersion();
  llvm_unreachable("Expecting OpenMP version attribute in module");
}

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

Fortran::semantics::Symbol *
getIterationVariableSymbol(const Fortran::lower::pft::Evaluation &eval) {
  return eval.visit(Fortran::common::visitors{
      [&](const Fortran::parser::DoConstruct &doLoop) {
        if (const auto &maybeCtrl = doLoop.GetLoopControl()) {
          using LoopControl = Fortran::parser::LoopControl;
          if (auto *bounds = std::get_if<LoopControl::Bounds>(&maybeCtrl->u)) {
            static_assert(
                std::is_same_v<decltype(bounds->name),
                               Fortran::parser::Scalar<Fortran::parser::Name>>);
            return bounds->name.thing.symbol;
          }
        }
        return static_cast<Fortran::semantics::Symbol *>(nullptr);
      },
      [](auto &&) {
        return static_cast<Fortran::semantics::Symbol *>(nullptr);
      },
  });
}

void gatherFuncAndVarSyms(
    const ObjectList &objects, mlir::omp::DeclareTargetCaptureClause clause,
    llvm::SmallVectorImpl<DeclareTargetCapturePair> &symbolAndClause) {
  for (const Object &object : objects)
    symbolAndClause.emplace_back(clause, *object.id());
}

mlir::omp::MapInfoOp
createMapInfoOp(fir::FirOpBuilder &builder, mlir::Location loc,
                mlir::Value baseAddr, mlir::Value varPtrPtr, std::string name,
                llvm::ArrayRef<mlir::Value> bounds,
                llvm::ArrayRef<mlir::Value> members,
                mlir::DenseIntElementsAttr membersIndex, uint64_t mapType,
                mlir::omp::VariableCaptureKind mapCaptureType, mlir::Type retTy,
                bool partialMap) {
  if (auto boxTy = baseAddr.getType().dyn_cast<fir::BaseBoxType>()) {
    baseAddr = builder.create<fir::BoxAddrOp>(loc, baseAddr);
    retTy = baseAddr.getType();
  }

  mlir::TypeAttr varType = mlir::TypeAttr::get(
      llvm::cast<mlir::omp::PointerLikeType>(retTy).getElementType());

  mlir::omp::MapInfoOp op = builder.create<mlir::omp::MapInfoOp>(
      loc, retTy, baseAddr, varType, varPtrPtr, members, membersIndex, bounds,
      builder.getIntegerAttr(builder.getIntegerType(64, false), mapType),
      builder.getAttr<mlir::omp::VariableCaptureKindAttr>(mapCaptureType),
      builder.getStringAttr(name), builder.getBoolAttr(partialMap));

  return op;
}

static int
getComponentPlacementInParent(const Fortran::semantics::Symbol *componentSym) {
  const auto *derived =
      componentSym->owner()
          .derivedTypeSpec()
          ->typeSymbol()
          .detailsIf<Fortran::semantics::DerivedTypeDetails>();
  assert(derived &&
         "expected derived type details when processing component symbol");
  for (auto [placement, name] : llvm::enumerate(derived->componentNames()))
    if (name == componentSym->name())
      return placement;
  return -1;
}

static std::optional<Object>
getComponentObject(std::optional<Object> object,
                   Fortran::semantics::SemanticsContext &semaCtx) {
  if (!object)
    return std::nullopt;

  auto ref = evaluate::ExtractDataRef(*object.value().ref());
  if (!ref)
    return std::nullopt;

  if (std::holds_alternative<evaluate::Component>(ref->u))
    return object;

  auto baseObj = getBaseObject(object.value(), semaCtx);
  if (!baseObj)
    return std::nullopt;

  return getComponentObject(baseObj.value(), semaCtx);
}

static void
generateMemberPlacementIndices(const Object &object,
                               llvm::SmallVectorImpl<int> &indices,
                               Fortran::semantics::SemanticsContext &semaCtx) {
  auto compObj = getComponentObject(object, semaCtx);
  while (compObj) {
    indices.push_back(getComponentPlacementInParent(compObj->id()));
    compObj =
        getComponentObject(getBaseObject(compObj.value(), semaCtx), semaCtx);
  }

  indices = llvm::SmallVector<int>{llvm::reverse(indices)};
}

void addChildIndexAndMapToParent(
    const omp::Object &object,
    std::map<const Fortran::semantics::Symbol *,
             llvm::SmallVector<OmpMapMemberIndicesData>> &parentMemberIndices,
    mlir::omp::MapInfoOp &mapOp,
    Fortran::semantics::SemanticsContext &semaCtx) {
  std::optional<Fortran::evaluate::DataRef> dataRef =
      ExtractDataRef(object.designator);
  assert(dataRef.has_value() &&
         "DataRef could not be extracted during mapping of derived type "
         "cannot proceed");
  const Fortran::semantics::Symbol *parentSym = &dataRef->GetFirstSymbol();
  assert(parentSym && "Could not find parent symbol during lower of "
                      "a component member in OpenMP map clause");
  llvm::SmallVector<int> indices;
  generateMemberPlacementIndices(object, indices, semaCtx);
  parentMemberIndices[parentSym].push_back({indices, mapOp});
}

static void calculateShapeAndFillIndices(
    llvm::SmallVectorImpl<int64_t> &shape,
    llvm::SmallVectorImpl<OmpMapMemberIndicesData> &memberPlacementData) {
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
  // index lists have to be of the same length, this emplaces -1 as filler.
  for (auto &v : memberPlacementData) {
    if (v.memberPlacementIndices.size() < largestIndicesSize) {
      auto *prevEnd = v.memberPlacementIndices.end();
      v.memberPlacementIndices.resize(largestIndicesSize);
      std::fill(prevEnd, v.memberPlacementIndices.end(), -1);
    }
  }
}

static mlir::DenseIntElementsAttr createDenseElementsAttrFromIndices(
    llvm::SmallVectorImpl<OmpMapMemberIndicesData> &memberPlacementData,
    fir::FirOpBuilder &builder) {
  llvm::SmallVector<int64_t> shape;
  calculateShapeAndFillIndices(shape, memberPlacementData);

  llvm::SmallVector<int> indicesFlattened = std::accumulate(
      memberPlacementData.begin(), memberPlacementData.end(),
      llvm::SmallVector<int>(),
      [](llvm::SmallVector<int> &x, OmpMapMemberIndicesData y) {
        x.insert(x.end(), y.memberPlacementIndices.begin(),
                 y.memberPlacementIndices.end());
        return x;
      });

  return mlir::DenseIntElementsAttr::get(
      mlir::VectorType::get(shape,
                            mlir::IntegerType::get(builder.getContext(), 32)),
      indicesFlattened);
}

void insertChildMapInfoIntoParent(
    Fortran::lower::AbstractConverter &converter,
    std::map<const Fortran::semantics::Symbol *,
             llvm::SmallVector<OmpMapMemberIndicesData>> &parentMemberIndices,
    llvm::SmallVectorImpl<mlir::Value> &mapOperands,
    llvm::SmallVectorImpl<const Fortran::semantics::Symbol *> &mapSyms,
    llvm::SmallVectorImpl<mlir::Type> *mapSymTypes,
    llvm::SmallVectorImpl<mlir::Location> *mapSymLocs) {
  for (auto indices : parentMemberIndices) {
    bool parentExists = false;
    size_t parentIdx;
    for (parentIdx = 0; parentIdx < mapSyms.size(); ++parentIdx) {
      if (mapSyms[parentIdx] == indices.first) {
        parentExists = true;
        break;
      }
    }

    if (parentExists) {
      auto mapOp = llvm::cast<mlir::omp::MapInfoOp>(
          mapOperands[parentIdx].getDefiningOp());

      // NOTE: To maintain appropriate SSA ordering, we move the parent map
      // which will now have references to its children after the last
      // of its members to be generated. This is necessary when a user
      // has defined a series of parent and children maps where the parent
      // precedes the children. An alternative, may be to do
      // delayed generation of map info operations from the clauses and
      // organize them first before generation.
      mapOp->moveAfter(indices.second.back().memberMap);

      for (auto memberIndicesData : indices.second)
        mapOp.getMembersMutable().append(
            memberIndicesData.memberMap.getResult());

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
      mlir::Value origSymbol = converter.getSymbolAddress(*indices.first);

      llvm::SmallVector<mlir::Value> members;
      for (OmpMapMemberIndicesData memberIndicesData : indices.second)
        members.push_back((mlir::Value)memberIndicesData.memberMap);

      mlir::Value mapOp = createMapInfoOp(
          converter.getFirOpBuilder(), origSymbol.getLoc(), origSymbol,
          /*varPtrPtr=*/mlir::Value(), indices.first->name().ToString(),
          /*bounds=*/{}, members,
          createDenseElementsAttrFromIndices(indices.second,
                                             converter.getFirOpBuilder()),
          mapType, mlir::omp::VariableCaptureKind::ByRef, origSymbol.getType(),
          /*partialMap=*/true);

      mapOperands.push_back(mapOp);
      mapSyms.push_back(indices.first);

      if (mapSymTypes)
        mapSymTypes->push_back(mapOp.getType());
      if (mapSymLocs)
        mapSymLocs->push_back(mapOp.getLoc());
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
              // Use getLastName to retrieve the arrays symbol, this will
              // provide the farthest right symbol (the last) in a designator,
              // i.e. providing something like the following:
              // "dtype1%dtype2%array[2:10]", will result in "array"
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

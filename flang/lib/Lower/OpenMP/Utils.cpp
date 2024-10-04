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
#include <DirectivesCommon.h>

#include <flang/Evaluate/fold.h>
#include <flang/Lower/AbstractConverter.h>
#include <flang/Lower/ConvertExprToHLFIR.h>
#include <flang/Lower/ConvertType.h>
#include <flang/Lower/PFTBuilder.h>
#include <flang/Lower/StatementContext.h>
#include <flang/Lower/SymbolMap.h>
#include <flang/Optimizer/Builder/FIRBuilder.h>
#include <flang/Optimizer/Builder/Todo.h>
#include <flang/Parser/parse-tree.h>
#include <flang/Parser/tools.h>
#include <flang/Semantics/tools.h>
#include <llvm/Support/CommandLine.h>

llvm::cl::opt<bool> treatIndexAsSection(
    "openmp-treat-index-as-section",
    llvm::cl::desc("In the OpenMP data clauses treat `a(N)` as `a(N:N)`."),
    llvm::cl::init(true));

llvm::cl::opt<bool> enableDelayedPrivatization(
    "openmp-enable-delayed-privatization",
    llvm::cl::desc(
        "Emit `[first]private` variables as clauses on the MLIR ops."),
    llvm::cl::init(true));

llvm::cl::opt<bool> enableDelayedPrivatizationStaging(
    "openmp-enable-delayed-privatization-staging",
    llvm::cl::desc("For partially supported constructs, emit `[first]private` "
                   "variables as clauses on the MLIR ops."),
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

void genObjectList(const ObjectList &objects,
                   lower::AbstractConverter &converter,
                   llvm::SmallVectorImpl<mlir::Value> &operands) {
  for (const Object &object : objects) {
    const semantics::Symbol *sym = object.sym();
    assert(sym && "Expected Symbol");
    if (mlir::Value variable = converter.getSymbolAddress(*sym)) {
      operands.push_back(variable);
    } else if (const auto *details =
                   sym->detailsIf<semantics::HostAssocDetails>()) {
      operands.push_back(converter.getSymbolAddress(details->symbol()));
      converter.copySymbolBinding(details->symbol(), *sym);
    }
  }
}

mlir::Type getLoopVarType(lower::AbstractConverter &converter,
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

semantics::Symbol *
getIterationVariableSymbol(const lower::pft::Evaluation &eval) {
  return eval.visit(common::visitors{
      [&](const parser::DoConstruct &doLoop) {
        if (const auto &maybeCtrl = doLoop.GetLoopControl()) {
          using LoopControl = parser::LoopControl;
          if (auto *bounds = std::get_if<LoopControl::Bounds>(&maybeCtrl->u)) {
            static_assert(std::is_same_v<decltype(bounds->name),
                                         parser::Scalar<parser::Name>>);
            return bounds->name.thing.symbol;
          }
        }
        return static_cast<semantics::Symbol *>(nullptr);
      },
      [](auto &&) { return static_cast<semantics::Symbol *>(nullptr); },
  });
}

void gatherFuncAndVarSyms(
    const ObjectList &objects, mlir::omp::DeclareTargetCaptureClause clause,
    llvm::SmallVectorImpl<DeclareTargetCapturePair> &symbolAndClause) {
  for (const Object &object : objects)
    symbolAndClause.emplace_back(clause, *object.sym());
}

mlir::omp::MapInfoOp createMapInfoOp(
    fir::FirOpBuilder &builder, mlir::Location loc, mlir::Value baseAddr,
    mlir::Value varPtrPtr, std::string name, mlir::ArrayRef<mlir::Value> bounds,
    mlir::ArrayRef<mlir::Value> members, mlir::ArrayAttr membersIndex,
    uint64_t mapType, mlir::omp::VariableCaptureKind mapCaptureType,
    mlir::Type retTy, bool partialMap) {
  if (auto boxTy = llvm::dyn_cast<fir::BaseBoxType>(baseAddr.getType())) {
    baseAddr = builder.create<fir::BoxAddrOp>(loc, baseAddr);
    retTy = baseAddr.getType();
  }

  mlir::TypeAttr varType = mlir::TypeAttr::get(
      llvm::cast<mlir::omp::PointerLikeType>(retTy).getElementType());

  // For types with unknown extents such as <2x?xi32> we discard the incomplete
  // type info and only retain the base type. The correct dimensions are later
  // recovered through the bounds info.
  if (auto seqType = llvm::dyn_cast<fir::SequenceType>(varType.getValue()))
    if (seqType.hasDynamicExtents())
      varType = mlir::TypeAttr::get(seqType.getEleTy());

  mlir::omp::MapInfoOp op = builder.create<mlir::omp::MapInfoOp>(
      loc, retTy, baseAddr, varType, varPtrPtr, members, membersIndex, bounds,
      builder.getIntegerAttr(builder.getIntegerType(64, false), mapType),
      builder.getAttr<mlir::omp::VariableCaptureKindAttr>(mapCaptureType),
      builder.getStringAttr(name), builder.getBoolAttr(partialMap));
  return op;
}

omp::ObjectList gatherObjects(omp::Object obj,
                              semantics::SemanticsContext &semaCtx) {
  omp::ObjectList objList;
  std::optional<omp::Object> baseObj = obj;
  while (baseObj.has_value()) {
    objList.push_back(baseObj.value());
    baseObj = getBaseObject(baseObj.value(), semaCtx);
  }
  return omp::ObjectList{llvm::reverse(objList)};
}

bool isDuplicateMemberMapInfo(OmpMapParentAndMemberData &parentMembers,
                              llvm::SmallVectorImpl<int64_t> &memberIndices) {
  for (auto memberData : parentMembers.memberPlacementIndices)
    if (std::equal(memberIndices.begin(), memberIndices.end(),
                   memberData.begin()))
      return true;
  return false;
}

static void generateArrayIndices(lower::AbstractConverter &converter,
                                 fir::FirOpBuilder &firOpBuilder,
                                 lower::StatementContext &stmtCtx,
                                 mlir::Location clauseLocation,
                                 llvm::SmallVectorImpl<mlir::Value> &indices,
                                 omp::Object object) {
  if (auto maybeRef = evaluate::ExtractDataRef(*object.ref())) {
    evaluate::DataRef ref = *maybeRef;
    if (auto *arr = std::get_if<evaluate::ArrayRef>(&ref.u)) {
      for (auto v : arr->subscript()) {
        if (std::holds_alternative<Triplet>(v.u)) {
          llvm_unreachable("Triplet indexing in map clause is unsupported");
        } else {
          auto expr =
              std::get<Fortran::evaluate::IndirectSubscriptIntegerExpr>(v.u);
          mlir::Value subscript = fir::getBase(
              converter.genExprValue(toEvExpr(expr.value()), stmtCtx));
          mlir::Value one = firOpBuilder.createIntegerConstant(
              clauseLocation, firOpBuilder.getIndexType(), 1);
          subscript = firOpBuilder.createConvert(
              clauseLocation, firOpBuilder.getIndexType(), subscript);
          indices.push_back(firOpBuilder.create<mlir::arith::SubIOp>(
              clauseLocation, subscript, one));
        }
      }
    }
  }
}

// When mapping members of derived types, there is a chance that one of the
// members along the way to a mapped member is an descriptor. In which case
// we have to make sure we generate a map for those along the way otherwise
// we will be missing a chunk of data required to actually map the member
// type to device. This function effectively generates these maps and the
// appropriate data accesses required to generate these maps. It will avoid
// creating duplicate maps, as duplicates are just as bad as unmapped
// descriptor data in a lot of cases for the runtime (and unnecessary
// data movement should be avoided where possible)
mlir::Value createParentSymAndGenIntermediateMaps(
    mlir::Location clauseLocation, lower::AbstractConverter &converter,
    semantics::SemanticsContext &semaCtx, lower::StatementContext &stmtCtx,
    omp::ObjectList &objectList, llvm::SmallVector<int64_t> &indices,
    OmpMapParentAndMemberData &parentMemberIndices, std::string asFortran,
    llvm::omp::OpenMPOffloadMappingFlags mapTypeBits) {

  auto arrayExprWithSubscript = [](omp::Object obj) {
    if (auto maybeRef = evaluate::ExtractDataRef(*obj.ref())) {
      evaluate::DataRef ref = *maybeRef;
      if (auto *arr = std::get_if<evaluate::ArrayRef>(&ref.u))
        return !arr->subscript().empty();
    }
    return false;
  };

  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
  lower::AddrAndBoundsInfo parentBaseAddr = lower::getDataOperandBaseAddr(
      converter, firOpBuilder, *objectList[0].sym(), clauseLocation);
  mlir::Value curValue = parentBaseAddr.addr;

  // Iterate over all objects in the objectList, this should consist of all
  // record types between the parent and the member being mapped (including
  // the parent). The object list may also contain array objects as well,
  // this can occur when specifying bounds or a specific element access
  // within a member map, we skip these.
  size_t currentIndex = 0;
  for (size_t i = 0; i < objectList.size(); ++i) {
    if (fir::SequenceType arrType = mlir::dyn_cast<fir::SequenceType>(
            fir::unwrapPassByRefType(curValue.getType()))) {
      if (arrayExprWithSubscript(objectList[i])) {
        llvm::SmallVector<mlir::Value> indices;
        generateArrayIndices(converter, firOpBuilder, stmtCtx, clauseLocation,
                             indices, objectList[i]);
        assert(!indices.empty() && "missing expected indices for map clause");
        curValue = firOpBuilder.create<fir::CoordinateOp>(
            clauseLocation, firOpBuilder.getRefType(arrType.getEleTy()),
            curValue, indices);
      }
    }

    if (fir::RecordType recordType = mlir::dyn_cast<fir::RecordType>(
            fir::unwrapPassByRefType(curValue.getType()))) {
      mlir::Value idxConst = firOpBuilder.createIntegerConstant(
          clauseLocation, firOpBuilder.getIndexType(), indices[currentIndex]);
      mlir::Type memberTy =
          recordType.getTypeList().at(indices[currentIndex]).second;
      curValue = firOpBuilder.create<fir::CoordinateOp>(
          clauseLocation, firOpBuilder.getRefType(memberTy), curValue,
          idxConst);

      if ((currentIndex == indices.size() - 1) ||
          !fir::isTypeWithDescriptor(memberTy)) {
        currentIndex++;
        continue;
      }

      llvm::SmallVector<int64_t> interimIndices(
          indices.begin(), std::next(indices.begin(), currentIndex + 1));
      if (!isDuplicateMemberMapInfo(parentMemberIndices, interimIndices)) {
        // Generate initial bounds operations using the standard lowering
        // utility
        llvm::SmallVector<mlir::Value> intermBounds;
        if (i + 1 < objectList.size() &&
            objectList[i + 1].sym()->IsObjectArray()) {
          std::stringstream intermFortran;
          Fortran::lower::gatherDataOperandAddrAndBounds<
              mlir::omp::MapBoundsOp, mlir::omp::MapBoundsType>(
              converter, converter.getFirOpBuilder(), semaCtx,
              converter.getFctCtx(), *objectList[i + 1].sym(),
              objectList[i + 1].ref(), clauseLocation, intermFortran,
              intermBounds, treatIndexAsSection);
        }

        llvm::omp::OpenMPOffloadMappingFlags intermMapType = mapTypeBits;
        // remove all map TO, FROM and TOFROM bits, from the intermediate
        // allocatable maps, we simply wish to alloc or release them. It may be
        // safer to just pass OMP_MAP_NONE as the map type, but we may still
        // need some of the other map types the mapped member utilises, so for
        // now it's good to keep an eye on this.
        intermMapType &= ~llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_TO;
        intermMapType &= ~llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_FROM;

        mlir::omp::MapInfoOp mapOp = createMapInfoOp(
            firOpBuilder, clauseLocation, curValue,
            /*varPtrPtr=*/mlir::Value{}, asFortran,
            /*bounds=*/intermBounds,
            /*members=*/{},
            /*membersIndex=*/mlir::ArrayAttr{},
            static_cast<
                std::underlying_type_t<llvm::omp::OpenMPOffloadMappingFlags>>(
                intermMapType),
            mlir::omp::VariableCaptureKind::ByRef, curValue.getType());

        parentMemberIndices.memberPlacementIndices.push_back(interimIndices);
        parentMemberIndices.memberMap.push_back(mapOp);
      }

      curValue = firOpBuilder.create<fir::LoadOp>(clauseLocation, curValue);
      currentIndex++;
    }
  }

  return curValue;
}

static int64_t
getComponentPlacementInParent(const semantics::Symbol *componentSym) {
  const auto *derived = componentSym->owner()
                            .derivedTypeSpec()
                            ->typeSymbol()
                            .detailsIf<semantics::DerivedTypeDetails>();
  assert(derived &&
         "expected derived type details when processing component symbol");
  for (auto [placement, name] : llvm::enumerate(derived->componentNames()))
    if (name == componentSym->name())
      return placement;
  return -1;
}

static std::optional<Object>
getComponentObject(std::optional<Object> object,
                   semantics::SemanticsContext &semaCtx) {
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

void generateMemberPlacementIndices(const Object &object,
                                    llvm::SmallVectorImpl<int64_t> &indices,
                                    semantics::SemanticsContext &semaCtx) {
  indices.clear();
  auto compObj = getComponentObject(object, semaCtx);

  while (compObj) {
    int64_t index = getComponentPlacementInParent(compObj->sym());
    assert(index >= 0);
    indices.push_back(index);
    compObj =
        getComponentObject(getBaseObject(compObj.value(), semaCtx), semaCtx);
  }

  indices = llvm::SmallVector<int64_t>{llvm::reverse(indices)};
}

void addChildIndexAndMapToParent(const omp::Object &object,
                                 OmpMapParentAndMemberData &parentMemberIndices,
                                 mlir::omp::MapInfoOp &mapOp,
                                 semantics::SemanticsContext &semaCtx) {
  llvm::SmallVector<int64_t> indices;
  generateMemberPlacementIndices(object, indices, semaCtx);
  parentMemberIndices.memberPlacementIndices.push_back(indices);
  parentMemberIndices.memberMap.push_back(mapOp);
}

bool isMemberOrParentAllocatableOrPointer(
    const Object &object, semantics::SemanticsContext &semaCtx) {
  if (semantics::IsAllocatableOrObjectPointer(object.sym()))
    return true;

  auto compObj = getBaseObject(object, semaCtx);
  while (compObj) {
    if (compObj.has_value() &&
        semantics::IsAllocatableOrObjectPointer(compObj.value().sym()))
      return true;
    compObj = getBaseObject(compObj.value(), semaCtx);
  }

  return false;
}

void insertChildMapInfoIntoParent(
    lower::AbstractConverter &converter, semantics::SemanticsContext &semaCtx,
    lower::StatementContext &stmtCtx,
    std::map<Object, OmpMapParentAndMemberData> &parentMemberIndices,
    llvm::SmallVectorImpl<mlir::Value> &mapOperands,
    llvm::SmallVectorImpl<mlir::Type> *mapSymTypes,
    llvm::SmallVectorImpl<mlir::Location> *mapSymLocs,
    llvm::SmallVectorImpl<const semantics::Symbol *> *mapSymbols) {

  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();

  for (auto indices : parentMemberIndices) {
    bool parentExists = false;
    size_t parentIdx;

    for (parentIdx = 0; parentIdx < mapSymbols->size(); ++parentIdx)
      if ((*mapSymbols)[parentIdx] == indices.first.sym()) {
        parentExists = true;
        break;
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
      // organize them first before generation. Or to use the
      // topologicalSort utility which will enforce a stronger SSA
      // dominance ordering at the cost of efficiency/time.
      mapOp->moveAfter(indices.second.memberMap.back());

      for (mlir::omp::MapInfoOp memberMap : indices.second.memberMap)
        mapOp.getMembersMutable().append(memberMap.getResult());

      mapOp.setMembersIndexAttr(firOpBuilder.create2DI64ArrayAttr(
          indices.second.memberPlacementIndices));
    } else {
      // NOTE: We take the map type of the first child, this may not
      // be the correct thing to do, however, we shall see. For the moment
      // it allows this to work with enter and exit without causing MLIR
      // verification issues. The more appropriate thing may be to take
      // the "main" map type clause from the directive being used.
      uint64_t mapType = indices.second.memberMap[0].getMapType().value_or(0);

      llvm::SmallVector<mlir::Value> members;
      for (mlir::omp::MapInfoOp memberMap : indices.second.memberMap)
        members.push_back(memberMap.getResult());

      // Create parent to emplace and bind members
      llvm::SmallVector<mlir::Value> bounds;
      std::stringstream asFortran;
      lower::AddrAndBoundsInfo info =
          lower::gatherDataOperandAddrAndBounds<mlir::omp::MapBoundsOp,
                                                mlir::omp::MapBoundsType>(
              converter, firOpBuilder, semaCtx, converter.getFctCtx(),
              *indices.first.sym(), indices.first.ref(),
              converter.getCurrentLocation(), asFortran, bounds,
              treatIndexAsSection);

      mlir::omp::MapInfoOp mapOp = createMapInfoOp(
          firOpBuilder, info.rawInput.getLoc(), info.rawInput,
          /*varPtrPtr=*/mlir::Value(), asFortran.str(), bounds, members,
          firOpBuilder.create2DI64ArrayAttr(
              indices.second.memberPlacementIndices),
          mapType, mlir::omp::VariableCaptureKind::ByRef,
          info.rawInput.getType(),
          /*partialMap=*/true);

      mapOperands.push_back(mapOp);

      if (mapSymTypes)
        mapSymTypes->push_back(mapOp.getType());
      if (mapSymLocs)
        mapSymLocs->push_back(mapOp.getLoc());
      if (mapSymbols)
        mapSymbols->push_back(indices.first.sym());
    }
  }
}

void lastprivateModifierNotSupported(const omp::clause::Lastprivate &lastp,
                                     mlir::Location loc) {
  using Lastprivate = omp::clause::Lastprivate;
  auto &maybeMod =
      std::get<std::optional<Lastprivate::LastprivateModifier>>(lastp.t);
  if (maybeMod) {
    assert(*maybeMod == Lastprivate::LastprivateModifier::Conditional &&
           "Unexpected lastprivate modifier");
    TODO(loc, "lastprivate clause with CONDITIONAL modifier");
  }
}

} // namespace omp
} // namespace lower
} // namespace Fortran

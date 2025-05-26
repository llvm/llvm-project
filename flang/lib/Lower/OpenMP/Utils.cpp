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

#include "ClauseFinder.h"
#include <flang/Lower/AbstractConverter.h>
#include <flang/Lower/ConvertType.h>
#include <flang/Lower/DirectivesCommon.h>
#include <flang/Lower/PFTBuilder.h>
#include <flang/Optimizer/Builder/FIRBuilder.h>
#include <flang/Optimizer/Builder/Todo.h>
#include <flang/Parser/parse-tree.h>
#include <flang/Parser/tools.h>
#include <flang/Semantics/tools.h>
#include <llvm/Support/CommandLine.h>

#include <iterator>

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

mlir::omp::MapInfoOp
createMapInfoOp(fir::FirOpBuilder &builder, mlir::Location loc,
                mlir::Value baseAddr, mlir::Value varPtrPtr,
                llvm::StringRef name, llvm::ArrayRef<mlir::Value> bounds,
                llvm::ArrayRef<mlir::Value> members,
                mlir::ArrayAttr membersIndex, uint64_t mapType,
                mlir::omp::VariableCaptureKind mapCaptureType, mlir::Type retTy,
                bool partialMap, mlir::FlatSymbolRefAttr mapperId) {
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
      loc, retTy, baseAddr, varType,
      builder.getIntegerAttr(builder.getIntegerType(64, false), mapType),
      builder.getAttr<mlir::omp::VariableCaptureKindAttr>(mapCaptureType),
      varPtrPtr, members, membersIndex, bounds, mapperId,
      builder.getStringAttr(name), builder.getBoolAttr(partialMap));
  return op;
}

// This function gathers the individual omp::Object's that make up a
// larger omp::Object symbol.
//
// For example, provided the larger symbol: "parent%child%member", this
// function breaks it up into its constituent components ("parent",
// "child", "member"), so we can access each individual component and
// introspect details. Important to note is this function breaks it up from
// RHS to LHS ("member" to "parent") and then we reverse it so that the
// returned omp::ObjectList is LHS to RHS, with the "parent" at the
// beginning.
omp::ObjectList gatherObjectsOf(omp::Object derivedTypeMember,
                                semantics::SemanticsContext &semaCtx) {
  omp::ObjectList objList;
  std::optional<omp::Object> baseObj = derivedTypeMember;
  while (baseObj.has_value()) {
    objList.push_back(baseObj.value());
    baseObj = getBaseObject(baseObj.value(), semaCtx);
  }
  return omp::ObjectList{llvm::reverse(objList)};
}

// This function generates a series of indices from a provided omp::Object,
// that devolves to an ArrayRef symbol, e.g. "array(2,3,4)", this function
// would generate a series of indices of "[1][2][3]" for the above example,
// offsetting by -1 to account for the non-zero fortran indexes.
//
// These indices can then be provided to a coordinate operation or other
// GEP-like operation to access the relevant positional member of the
// array.
//
// It is of note that the function only supports subscript integers currently
// and not Triplets i.e. Array(1:2:3).
static void generateArrayIndices(lower::AbstractConverter &converter,
                                 fir::FirOpBuilder &firOpBuilder,
                                 lower::StatementContext &stmtCtx,
                                 mlir::Location clauseLocation,
                                 llvm::SmallVectorImpl<mlir::Value> &indices,
                                 omp::Object object) {
  auto maybeRef = evaluate::ExtractDataRef(*object.ref());
  if (!maybeRef)
    return;

  auto *arr = std::get_if<evaluate::ArrayRef>(&maybeRef->u);
  if (!arr)
    return;

  for (auto v : arr->subscript()) {
    if (std::holds_alternative<Triplet>(v.u))
      TODO(clauseLocation, "Triplet indexing in map clause is unsupported");

    auto expr = std::get<Fortran::evaluate::IndirectSubscriptIntegerExpr>(v.u);
    mlir::Value subscript =
        fir::getBase(converter.genExprValue(toEvExpr(expr.value()), stmtCtx));
    mlir::Value one = firOpBuilder.createIntegerConstant(
        clauseLocation, firOpBuilder.getIndexType(), 1);
    subscript = firOpBuilder.createConvert(
        clauseLocation, firOpBuilder.getIndexType(), subscript);
    indices.push_back(firOpBuilder.create<mlir::arith::SubIOp>(clauseLocation,
                                                               subscript, one));
  }
}

/// When mapping members of derived types, there is a chance that one of the
/// members along the way to a mapped member is an descriptor. In which case
/// we have to make sure we generate a map for those along the way otherwise
/// we will be missing a chunk of data required to actually map the member
/// type to device. This function effectively generates these maps and the
/// appropriate data accesses required to generate these maps. It will avoid
/// creating duplicate maps, as duplicates are just as bad as unmapped
/// descriptor data in a lot of cases for the runtime (and unnecessary
/// data movement should be avoided where possible).
///
/// As an example for the following mapping:
///
/// type :: vertexes
///     integer(4), allocatable :: vertexx(:)
///     integer(4), allocatable :: vertexy(:)
/// end type vertexes
///
/// type :: dtype
///     real(4) :: i
///     type(vertexes), allocatable :: vertexes(:)
/// end type dtype
///
/// type(dtype), allocatable :: alloca_dtype
///
/// !$omp target map(tofrom: alloca_dtype%vertexes(N1)%vertexx)
///
/// The below HLFIR/FIR is generated (trimmed for conciseness):
///
/// On the first iteration we index into the record type alloca_dtype
/// to access "vertexes", we then generate a map for this descriptor
/// alongside bounds to indicate we only need the 1 member, rather than
/// the whole array block in this case (In theory we could map its
/// entirety at the cost of data transfer bandwidth).
///
/// %13:2 = hlfir.declare ... "alloca_dtype" ...
/// %39 = fir.load %13#0 : ...
/// %40 = fir.coordinate_of %39, %c1 : ...
/// %51 = omp.map.info var_ptr(%40 : ...) map_clauses(to) capture(ByRef) ...
/// %52 = fir.load %40 : ...
///
/// Second iteration generating access to "vertexes(N1) utilising the N1 index
/// %53 = load N1 ...
/// %54 = fir.convert %53 : (i32) -> i64
/// %55 = fir.convert %54 : (i64) -> index
/// %56 = arith.subi %55, %c1 : index
/// %57 = fir.coordinate_of %52, %56 : ...
///
/// Still in the second iteration we access the allocatable member "vertexx",
/// we return %58 from the function and provide it to the final and "main"
/// map of processMap (generated by the record type segment of the below
/// function), if this were not the final symbol in the list, i.e. we accessed
/// a member below vertexx, we would have generated the map below as we did in
/// the first iteration and then continue to generate further coordinates to
/// access further components as required.
///
/// %58 = fir.coordinate_of %57, %c0 : ...
/// %61 = omp.map.info var_ptr(%58 : ...) map_clauses(to) capture(ByRef) ...
///
/// Parent mapping containing prior generated mapped members, generated at
/// a later step but here to showcase the "end" result
///
/// omp.map.info var_ptr(%13#1 : ...) map_clauses(to) capture(ByRef)
///   members(%50, %61 : [0, 1, 0], [0, 1, 0] : ...
///
/// \param objectList - The list of omp::Object symbol data for each parent
///  to the mapped member (also includes the mapped member), generated via
///  gatherObjectsOf.
/// \param indices - List of index data associated with the mapped member
///   symbol, which identifies the placement of the member in its parent,
///   this helps generate the appropriate member accesses. These indices
///   can be generated via generateMemberPlacementIndices.
/// \param asFortran - A string generated from the mapped variable to be
///   associated with the main map, generally (but not restricted to)
///   generated via gatherDataOperandAddrAndBounds or other
///   DirectiveCommons.hpp utilities.
/// \param mapTypeBits - The map flags that will be associated with the
///   generated maps, minus alterations of the TO and FROM bits for the
///   intermediate components to prevent accidental overwriting on device
///   write back.
mlir::Value createParentSymAndGenIntermediateMaps(
    mlir::Location clauseLocation, lower::AbstractConverter &converter,
    semantics::SemanticsContext &semaCtx, lower::StatementContext &stmtCtx,
    omp::ObjectList &objectList, llvm::SmallVectorImpl<int64_t> &indices,
    OmpMapParentAndMemberData &parentMemberIndices, llvm::StringRef asFortran,
    llvm::omp::OpenMPOffloadMappingFlags mapTypeBits) {
  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();

  /// Checks if an omp::Object is an array expression with a subscript, e.g.
  /// array(1,2).
  auto isArrayExprWithSubscript = [](omp::Object obj) {
    if (auto maybeRef = evaluate::ExtractDataRef(obj.ref())) {
      evaluate::DataRef ref = *maybeRef;
      if (auto *arr = std::get_if<evaluate::ArrayRef>(&ref.u))
        return !arr->subscript().empty();
    }
    return false;
  };

  // Generate the access to the original parent base address.
  fir::factory::AddrAndBoundsInfo parentBaseAddr =
      lower::getDataOperandBaseAddr(converter, firOpBuilder,
                                    *objectList[0].sym(), clauseLocation);
  mlir::Value curValue = parentBaseAddr.addr;

  // Iterate over all objects in the objectList, this should consist of all
  // record types between the parent and the member being mapped (including
  // the parent). The object list may also contain array objects as well,
  // this can occur when specifying bounds or a specific element access
  // within a member map, we skip these.
  size_t currentIndicesIdx = 0;
  for (size_t i = 0; i < objectList.size(); ++i) {
    // If we encounter a sequence type, i.e. an array, we must generate the
    // correct coordinate operation to index into the array to proceed further,
    // this is only relevant in cases where we encounter subscripts currently.
    //
    // For example in the following case:
    //
    //   map(tofrom: array_dtype(4)%internal_dtypes(3)%float_elements(4))
    //
    // We must generate coordinate operation accesses for each subscript
    // we encounter.
    if (fir::SequenceType arrType = mlir::dyn_cast<fir::SequenceType>(
            fir::unwrapPassByRefType(curValue.getType()))) {
      if (isArrayExprWithSubscript(objectList[i])) {
        llvm::SmallVector<mlir::Value> subscriptIndices;
        generateArrayIndices(converter, firOpBuilder, stmtCtx, clauseLocation,
                             subscriptIndices, objectList[i]);
        assert(!subscriptIndices.empty() &&
               "missing expected indices for map clause");
        curValue = firOpBuilder.create<fir::CoordinateOp>(
            clauseLocation, firOpBuilder.getRefType(arrType.getEleTy()),
            curValue, subscriptIndices);
      }
    }

    // If we encounter a record type, we must access the subsequent member
    // by indexing into it and creating a coordinate operation to do so, we
    // utilise the index information generated previously and passed in to
    // work out the correct member to access and the corresponding member
    // type.
    if (fir::RecordType recordType = mlir::dyn_cast<fir::RecordType>(
            fir::unwrapPassByRefType(curValue.getType()))) {
      fir::IntOrValue idxConst = mlir::IntegerAttr::get(
          firOpBuilder.getI32Type(), indices[currentIndicesIdx]);
      mlir::Type memberTy = recordType.getType(indices[currentIndicesIdx]);
      curValue = firOpBuilder.create<fir::CoordinateOp>(
          clauseLocation, firOpBuilder.getRefType(memberTy), curValue,
          llvm::SmallVector<fir::IntOrValue, 1>{idxConst});

      // If we're a final member, the map will be generated by the processMap
      // call that invoked this function.
      if (currentIndicesIdx == indices.size() - 1)
        break;

      // Skip mapping and the subsequent load if we're not
      // a type with a descriptor such as a pointer/allocatable. If we're not a
      // type with a descriptor then we have no need of generating an
      // intermediate map for it, as we only need to generate a map if a member
      // is a descriptor type (and thus obscures the members it contains via a
      // pointer in which it's data needs mapped).
      if (!fir::isTypeWithDescriptor(memberTy)) {
        currentIndicesIdx++;
        continue;
      }

      llvm::SmallVector<int64_t> interimIndices(
          indices.begin(), std::next(indices.begin(), currentIndicesIdx + 1));
      // Verify we haven't already created a map for this particular member, by
      // checking the list of members already mapped for the current parent,
      // stored in the parentMemberIndices structure
      if (!parentMemberIndices.isDuplicateMemberMapInfo(interimIndices)) {
        // Generate bounds operations using the standard lowering utility,
        // unfortunately this currently does a bit more than just generate
        // bounds and we discard the other bits. May be useful to extend the
        // utility to just provide bounds in the future.
        llvm::SmallVector<mlir::Value> interimBounds;
        if (i + 1 < objectList.size() &&
            objectList[i + 1].sym()->IsObjectArray()) {
          std::stringstream interimFortran;
          Fortran::lower::gatherDataOperandAddrAndBounds<
              mlir::omp::MapBoundsOp, mlir::omp::MapBoundsType>(
              converter, converter.getFirOpBuilder(), semaCtx,
              converter.getFctCtx(), *objectList[i + 1].sym(),
              objectList[i + 1].ref(), clauseLocation, interimFortran,
              interimBounds, treatIndexAsSection);
        }

        // Remove all map-type bits (e.g. TO, FROM, etc.) from the intermediate
        // allocatable maps, as we simply wish to alloc or release them. It may
        // be safer to just pass OMP_MAP_NONE as the map type, but we may still
        // need some of the other map types the mapped member utilises, so for
        // now it's good to keep an eye on this.
        llvm::omp::OpenMPOffloadMappingFlags interimMapType = mapTypeBits;
        interimMapType &= ~llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_TO;
        interimMapType &= ~llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_FROM;
        interimMapType &=
            ~llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_RETURN_PARAM;

        // Create a map for the intermediate member and insert it and it's
        // indices into the parentMemberIndices list to track it.
        mlir::omp::MapInfoOp mapOp = createMapInfoOp(
            firOpBuilder, clauseLocation, curValue,
            /*varPtrPtr=*/mlir::Value{}, asFortran,
            /*bounds=*/interimBounds,
            /*members=*/{},
            /*membersIndex=*/mlir::ArrayAttr{},
            static_cast<
                std::underlying_type_t<llvm::omp::OpenMPOffloadMappingFlags>>(
                interimMapType),
            mlir::omp::VariableCaptureKind::ByRef, curValue.getType());

        parentMemberIndices.memberPlacementIndices.push_back(interimIndices);
        parentMemberIndices.memberMap.push_back(mapOp);
      }

      // Load the currently accessed member, so we can continue to access
      // further segments.
      curValue = firOpBuilder.create<fir::LoadOp>(clauseLocation, curValue);
      currentIndicesIdx++;
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

  auto ref = evaluate::ExtractDataRef(object.value().ref());
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
  assert(indices.empty() && "indices vector passed to "
                            "generateMemberPlacementIndices should be empty");
  auto compObj = getComponentObject(object, semaCtx);

  while (compObj) {
    int64_t index = getComponentPlacementInParent(compObj->sym());
    assert(
        index >= 0 &&
        "unexpected index value returned from getComponentPlacementInParent");
    indices.push_back(index);
    compObj =
        getComponentObject(getBaseObject(compObj.value(), semaCtx), semaCtx);
  }

  indices = llvm::SmallVector<int64_t>{llvm::reverse(indices)};
}

void OmpMapParentAndMemberData::addChildIndexAndMapToParent(
    const omp::Object &object, mlir::omp::MapInfoOp &mapOp,
    semantics::SemanticsContext &semaCtx) {
  llvm::SmallVector<int64_t> indices;
  generateMemberPlacementIndices(object, indices, semaCtx);
  memberPlacementIndices.push_back(indices);
  memberMap.push_back(mapOp);
}

bool isMemberOrParentAllocatableOrPointer(
    const Object &object, semantics::SemanticsContext &semaCtx) {
  if (semantics::IsAllocatableOrObjectPointer(object.sym()))
    return true;

  auto compObj = getBaseObject(object, semaCtx);
  while (compObj) {
    if (semantics::IsAllocatableOrObjectPointer(compObj.value().sym()))
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
    llvm::SmallVectorImpl<const semantics::Symbol *> &mapSyms) {
  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
  for (auto indices : parentMemberIndices) {
    auto *parentIter =
        llvm::find_if(mapSyms, [&indices](const semantics::Symbol *v) {
          return v == indices.first.sym();
        });
    if (parentIter != mapSyms.end()) {
      auto mapOp = llvm::cast<mlir::omp::MapInfoOp>(
          mapOperands[std::distance(mapSyms.begin(), parentIter)]
              .getDefiningOp());

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
      uint64_t mapType = indices.second.memberMap[0].getMapType();

      llvm::SmallVector<mlir::Value> members;
      members.reserve(indices.second.memberMap.size());
      for (mlir::omp::MapInfoOp memberMap : indices.second.memberMap)
        members.push_back(memberMap.getResult());

      // Create parent to emplace and bind members
      llvm::SmallVector<mlir::Value> bounds;
      std::stringstream asFortran;
      fir::factory::AddrAndBoundsInfo info =
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
      mapSyms.push_back(indices.first.sym());
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

static void convertLoopBounds(lower::AbstractConverter &converter,
                              mlir::Location loc,
                              mlir::omp::LoopRelatedClauseOps &result,
                              std::size_t loopVarTypeSize) {
  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
  // The types of lower bound, upper bound, and step are converted into the
  // type of the loop variable if necessary.
  mlir::Type loopVarType = getLoopVarType(converter, loopVarTypeSize);
  for (unsigned it = 0; it < (unsigned)result.loopLowerBounds.size(); it++) {
    result.loopLowerBounds[it] = firOpBuilder.createConvert(
        loc, loopVarType, result.loopLowerBounds[it]);
    result.loopUpperBounds[it] = firOpBuilder.createConvert(
        loc, loopVarType, result.loopUpperBounds[it]);
    result.loopSteps[it] =
        firOpBuilder.createConvert(loc, loopVarType, result.loopSteps[it]);
  }
}

bool collectLoopRelatedInfo(
    lower::AbstractConverter &converter, mlir::Location currentLocation,
    lower::pft::Evaluation &eval, const omp::List<omp::Clause> &clauses,
    mlir::omp::LoopRelatedClauseOps &result,
    llvm::SmallVectorImpl<const semantics::Symbol *> &iv) {
  bool found = false;
  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();

  // Collect the loops to collapse.
  lower::pft::Evaluation *doConstructEval = &eval.getFirstNestedEvaluation();
  if (doConstructEval->getIf<parser::DoConstruct>()->IsDoConcurrent()) {
    TODO(currentLocation, "Do Concurrent in Worksharing loop construct");
  }

  std::int64_t collapseValue = 1l;
  if (auto *clause =
          ClauseFinder::findUniqueClause<omp::clause::Collapse>(clauses)) {
    collapseValue = evaluate::ToInt64(clause->v).value();
    found = true;
  }

  std::size_t loopVarTypeSize = 0;
  do {
    lower::pft::Evaluation *doLoop =
        &doConstructEval->getFirstNestedEvaluation();
    auto *doStmt = doLoop->getIf<parser::NonLabelDoStmt>();
    assert(doStmt && "Expected do loop to be in the nested evaluation");
    const auto &loopControl =
        std::get<std::optional<parser::LoopControl>>(doStmt->t);
    const parser::LoopControl::Bounds *bounds =
        std::get_if<parser::LoopControl::Bounds>(&loopControl->u);
    assert(bounds && "Expected bounds for worksharing do loop");
    lower::StatementContext stmtCtx;
    result.loopLowerBounds.push_back(fir::getBase(
        converter.genExprValue(*semantics::GetExpr(bounds->lower), stmtCtx)));
    result.loopUpperBounds.push_back(fir::getBase(
        converter.genExprValue(*semantics::GetExpr(bounds->upper), stmtCtx)));
    if (bounds->step) {
      result.loopSteps.push_back(fir::getBase(
          converter.genExprValue(*semantics::GetExpr(bounds->step), stmtCtx)));
    } else { // If `step` is not present, assume it as `1`.
      result.loopSteps.push_back(firOpBuilder.createIntegerConstant(
          currentLocation, firOpBuilder.getIntegerType(32), 1));
    }
    iv.push_back(bounds->name.thing.symbol);
    loopVarTypeSize = std::max(loopVarTypeSize,
                               bounds->name.thing.symbol->GetUltimate().size());
    collapseValue--;
    doConstructEval =
        &*std::next(doConstructEval->getNestedEvaluations().begin());
  } while (collapseValue > 0);

  convertLoopBounds(converter, currentLocation, result, loopVarTypeSize);

  return found;
}
} // namespace omp
} // namespace lower
} // namespace Fortran

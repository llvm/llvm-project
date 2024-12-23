//===- MapInfoFinalization.cpp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
/// \file
/// An OpenMP dialect related pass for FIR/HLFIR which performs some
/// pre-processing of MapInfoOp's after the module has been lowered to
/// finalize them.
///
/// For example, it expands MapInfoOp's containing descriptor related
/// types (fir::BoxType's) into multiple MapInfoOp's containing the parent
/// descriptor and pointer member components for individual mapping,
/// treating the descriptor type as a record type for later lowering in the
/// OpenMP dialect.
///
/// The pass also adds MapInfoOp's that are members of a parent object but are
/// not directly used in the body of a target region to its BlockArgument list
/// to maintain consistency across all MapInfoOp's tied to a region directly or
/// indirectly via a parent object.
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Builder/DirectivesCommon.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/HLFIRTools.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Dialect/Support/KindMapping.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "flang/Optimizer/OpenMP/Passes.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Frontend/OpenMP/OMPConstants.h"
#include <algorithm>
#include <cstddef>
#include <iterator>
#include <numeric>

namespace flangomp {
#define GEN_PASS_DEF_MAPINFOFINALIZATIONPASS
#include "flang/Optimizer/OpenMP/Passes.h.inc"
} // namespace flangomp

namespace {
class MapInfoFinalizationPass
    : public flangomp::impl::MapInfoFinalizationPassBase<
          MapInfoFinalizationPass> {
  /// Helper class tracking a members parent and its
  /// placement in the parents member list
  struct ParentAndPlacement {
    mlir::omp::MapInfoOp parent;
    size_t index;
  };

  /// Tracks any intermediate function/subroutine local allocations we
  /// generate for the descriptors of box type dummy arguments, so that
  /// we can retrieve it for subsequent reuses within the functions
  /// scope.
  ///
  ///      descriptor defining op
  ///      |                  corresponding local alloca
  ///      |                  |
  std::map<mlir::Operation *, mlir::Value> localBoxAllocas;

  /// getMemberUserList gathers all users of a particular MapInfoOp that are
  /// other MapInfoOp's and places them into the mapMemberUsers list, which
  /// records the map that the current argument MapInfoOp "op" is part of
  /// alongside the placement of "op" in the recorded users members list. The
  /// intent of the generated list is to find all MapInfoOp's that may be
  /// considered parents of the passed in "op" and in which it shows up in the
  /// member list, alongside collecting the placement information of "op" in its
  /// parents member list.
  void
  getMemberUserList(mlir::omp::MapInfoOp op,
                    llvm::SmallVectorImpl<ParentAndPlacement> &mapMemberUsers) {
    for (auto *user : op->getUsers())
      if (auto map = mlir::dyn_cast_if_present<mlir::omp::MapInfoOp>(user))
        for (auto [i, mapMember] : llvm::enumerate(map.getMembers()))
          if (mapMember.getDefiningOp() == op)
            mapMemberUsers.push_back({map, i});
  }

  void getAsIntegers(llvm::ArrayRef<mlir::Attribute> values,
                     llvm::SmallVectorImpl<int64_t> &ints) {
    ints.reserve(values.size());
    llvm::transform(values, std::back_inserter(ints),
                    [](mlir::Attribute value) {
                      return mlir::cast<mlir::IntegerAttr>(value).getInt();
                    });
  }

  /// This function will expand a MapInfoOp's member indices back into a vector
  /// so that they can be trivially modified as unfortunately the attribute type
  /// that's used does not have modifiable fields at the moment (generally
  /// awkward to work with)
  void getMemberIndicesAsVectors(
      mlir::omp::MapInfoOp mapInfo,
      llvm::SmallVectorImpl<llvm::SmallVector<int64_t>> &indices) {
    indices.reserve(mapInfo.getMembersIndexAttr().getValue().size());
    llvm::transform(mapInfo.getMembersIndexAttr().getValue(),
                    std::back_inserter(indices), [this](mlir::Attribute value) {
                      auto memberIndex = mlir::cast<mlir::ArrayAttr>(value);
                      llvm::SmallVector<int64_t> indexes;
                      getAsIntegers(memberIndex.getValue(), indexes);
                      return indexes;
                    });
  }

  /// When provided a MapInfoOp containing a descriptor type that
  /// we must expand into multiple maps this function will extract
  /// the value from it and return it, in certain cases we must
  /// generate a new allocation to store into so that the
  /// fir::BoxOffsetOp we utilise to access the descriptor datas
  /// base address can be utilised.
  mlir::Value getDescriptorFromBoxMap(mlir::omp::MapInfoOp boxMap,
                                      fir::FirOpBuilder &builder) {
    mlir::Value descriptor = boxMap.getVarPtr();
    if (!fir::isTypeWithDescriptor(boxMap.getVarType()))
      if (auto addrOp = mlir::dyn_cast_if_present<fir::BoxAddrOp>(
              boxMap.getVarPtr().getDefiningOp()))
        descriptor = addrOp.getVal();

    if (!mlir::isa<fir::BaseBoxType>(descriptor.getType()))
      return descriptor;

    mlir::Value &slot = localBoxAllocas[descriptor.getDefiningOp()];
    if (slot) {
      return slot;
    }

    // The fir::BoxOffsetOp only works with !fir.ref<!fir.box<...>> types, as
    // allowing it to access non-reference box operations can cause some
    // problematic SSA IR. However, in the case of assumed shape's the type
    // is not a !fir.ref, in these cases to retrieve the appropriate
    // !fir.ref<!fir.box<...>> to access the data we need to map we must
    // perform an alloca and then store to it and retrieve the data from the new
    // alloca.
    mlir::OpBuilder::InsertPoint insPt = builder.saveInsertionPoint();
    mlir::Block *allocaBlock = builder.getAllocaBlock();
    mlir::Location loc = boxMap->getLoc();
    assert(allocaBlock && "No alloca block found for this top level op");
    builder.setInsertionPointToStart(allocaBlock);
    auto alloca = builder.create<fir::AllocaOp>(loc, descriptor.getType());
    builder.restoreInsertionPoint(insPt);
    builder.create<fir::StoreOp>(loc, descriptor, alloca);
    return slot = alloca;
  }

  /// Function that generates a FIR operation accessing the descriptor's
  /// base address (BoxOffsetOp) and a MapInfoOp for it. The most
  /// important thing to note is that we normally move the bounds from
  /// the descriptor map onto the base address map.
  mlir::omp::MapInfoOp genBaseAddrMap(mlir::Value descriptor,
                                      mlir::OperandRange bounds,
                                      int64_t mapType,
                                      fir::FirOpBuilder &builder) {
    mlir::Location loc = descriptor.getLoc();
    mlir::Value baseAddrAddr = builder.create<fir::BoxOffsetOp>(
        loc, descriptor, fir::BoxFieldAttr::base_addr);

    mlir::Type underlyingVarType =
        llvm::cast<mlir::omp::PointerLikeType>(
            fir::unwrapRefType(baseAddrAddr.getType()))
            .getElementType();
    if (auto seqType = llvm::dyn_cast<fir::SequenceType>(underlyingVarType))
      if (seqType.hasDynamicExtents())
        underlyingVarType = seqType.getEleTy();

    // Member of the descriptor pointing at the allocated data
    return builder.create<mlir::omp::MapInfoOp>(
        loc, baseAddrAddr.getType(), descriptor,
        mlir::TypeAttr::get(underlyingVarType), baseAddrAddr,
        /*members=*/mlir::SmallVector<mlir::Value>{},
        /*membersIndex=*/mlir::ArrayAttr{}, bounds,
        builder.getIntegerAttr(builder.getIntegerType(64, false), mapType),
        builder.getAttr<mlir::omp::VariableCaptureKindAttr>(
            mlir::omp::VariableCaptureKind::ByRef),
        /*name=*/builder.getStringAttr(""),
        /*partial_map=*/builder.getBoolAttr(false));
  }

  /// This function adjusts the member indices vector to include a new
  /// base address member. We take the position of the descriptor in
  /// the member indices list, which is the index data that the base
  /// addresses index will be based off of, as the base address is
  /// a member of the descriptor. We must also alter other members
  /// that are members of this descriptor to account for the addition
  /// of the base address index.
  void adjustMemberIndices(
      llvm::SmallVectorImpl<llvm::SmallVector<int64_t>> &memberIndices,
      size_t memberIndex) {
    llvm::SmallVector<int64_t> baseAddrIndex = memberIndices[memberIndex];

    // If we find another member that is "derived/a member of" the descriptor
    // that is not the descriptor itself, we must insert a 0 for the new base
    // address we have just added for the descriptor into the list at the
    // appropriate position to maintain correctness of the positional/index data
    // for that member.
    for (llvm::SmallVector<int64_t> &member : memberIndices)
      if (member.size() > baseAddrIndex.size() &&
          std::equal(baseAddrIndex.begin(), baseAddrIndex.end(),
                     member.begin()))
        member.insert(std::next(member.begin(), baseAddrIndex.size()), 0);

    // Add the base address index to the main base address member data
    baseAddrIndex.push_back(0);

    // Insert our newly created baseAddrIndex into the larger list of indices at
    // the correct location.
    memberIndices.insert(std::next(memberIndices.begin(), memberIndex + 1),
                         baseAddrIndex);
  }

  /// Adjusts the descriptor's map type. The main alteration that is done
  /// currently is transforming the map type to `OMP_MAP_TO` where possible.
  /// This is because we will always need to map the descriptor to device
  /// (or at the very least it seems to be the case currently with the
  /// current lowered kernel IR), as without the appropriate descriptor
  /// information on the device there is a risk of the kernel IR
  /// requesting for various data that will not have been copied to
  /// perform things like indexing. This can cause segfaults and
  /// memory access errors. However, we do not need this data mapped
  /// back to the host from the device, as per the OpenMP spec we cannot alter
  /// the data via resizing or deletion on the device. Discarding any
  /// descriptor alterations via no map back is reasonable (and required
  /// for certain segments of descriptor data like the type descriptor that are
  /// global constants). This alteration is only inapplicable to `target exit`
  /// and `target update` currently, and that's due to `target exit` not
  /// allowing `to` mappings, and `target update` not allowing both `to` and
  /// `from` simultaneously. We currently try to maintain the `implicit` flag
  /// where necessary, although it does not seem strictly required.
  unsigned long getDescriptorMapType(unsigned long mapTypeFlag,
                                     mlir::Operation *target) {
    if (llvm::isa_and_nonnull<mlir::omp::TargetExitDataOp,
                              mlir::omp::TargetUpdateOp>(target))
      return mapTypeFlag;

    bool hasImplicitMap =
        (llvm::omp::OpenMPOffloadMappingFlags(mapTypeFlag) &
         llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_IMPLICIT) ==
        llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_IMPLICIT;

    return llvm::to_underlying(
        hasImplicitMap
            ? llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_TO |
                  llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_IMPLICIT
            : llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_TO);
  }

  mlir::omp::MapInfoOp genDescriptorMemberMaps(mlir::omp::MapInfoOp op,
                                               fir::FirOpBuilder &builder,
                                               mlir::Operation *target) {
    llvm::SmallVector<ParentAndPlacement> mapMemberUsers;
    getMemberUserList(op, mapMemberUsers);

    // TODO: map the addendum segment of the descriptor, similarly to the
    // base address/data pointer member.
    mlir::Value descriptor = getDescriptorFromBoxMap(op, builder);
    auto baseAddr = genBaseAddrMap(descriptor, op.getBounds(),
                                   op.getMapType().value_or(0), builder);
    mlir::ArrayAttr newMembersAttr;
    mlir::SmallVector<mlir::Value> newMembers;
    llvm::SmallVector<llvm::SmallVector<int64_t>> memberIndices;

    if (!mapMemberUsers.empty() || !op.getMembers().empty())
      getMemberIndicesAsVectors(
          !mapMemberUsers.empty() ? mapMemberUsers[0].parent : op,
          memberIndices);

    // If the operation that we are expanding with a descriptor has a user
    // (parent), then we have to expand the parent's member indices to reflect
    // the adjusted member indices for the base address insertion. However, if
    // it does not then we are expanding a MapInfoOp without any pre-existing
    // member information to now have one new member for the base address, or
    // we are expanding a parent that is a descriptor and we have to adjust
    // all of its members to reflect the insertion of the base address.
    if (!mapMemberUsers.empty()) {
      // Currently, there should only be one user per map when this pass
      // is executed. Either a parent map, holding the current map in its
      // member list, or a target operation that holds a map clause. This
      // may change in the future if we aim to refactor the MLIR for map
      // clauses to allow sharing of duplicate maps across target
      // operations.
      assert(mapMemberUsers.size() == 1 &&
             "OMPMapInfoFinalization currently only supports single users of a "
             "MapInfoOp");
      ParentAndPlacement mapUser = mapMemberUsers[0];
      adjustMemberIndices(memberIndices, mapUser.index);
      llvm::SmallVector<mlir::Value> newMemberOps;
      for (auto v : mapUser.parent.getMembers()) {
        newMemberOps.push_back(v);
        if (v == op)
          newMemberOps.push_back(baseAddr);
      }
      mapUser.parent.getMembersMutable().assign(newMemberOps);
      mapUser.parent.setMembersIndexAttr(
          builder.create2DI64ArrayAttr(memberIndices));
    } else {
      newMembers.push_back(baseAddr);
      if (!op.getMembers().empty()) {
        for (auto &indices : memberIndices)
          indices.insert(indices.begin(), 0);
        memberIndices.insert(memberIndices.begin(), {0});
        newMembersAttr = builder.create2DI64ArrayAttr(memberIndices);
        newMembers.append(op.getMembers().begin(), op.getMembers().end());
      } else {
        llvm::SmallVector<llvm::SmallVector<int64_t>> memberIdx = {{0}};
        newMembersAttr = builder.create2DI64ArrayAttr(memberIdx);
      }
    }

    mlir::omp::MapInfoOp newDescParentMapOp =
        builder.create<mlir::omp::MapInfoOp>(
            op->getLoc(), op.getResult().getType(), descriptor,
            mlir::TypeAttr::get(fir::unwrapRefType(descriptor.getType())),
            /*varPtrPtr=*/mlir::Value{}, newMembers, newMembersAttr,
            /*bounds=*/mlir::SmallVector<mlir::Value>{},
            builder.getIntegerAttr(
                builder.getIntegerType(64, false),
                getDescriptorMapType(op.getMapType().value_or(0), target)),
            op.getMapCaptureTypeAttr(), op.getNameAttr(),
            /*partial_map=*/builder.getBoolAttr(false));
    op.replaceAllUsesWith(newDescParentMapOp.getResult());
    op->erase();
    return newDescParentMapOp;
  }

  // We add all mapped record members not directly used in the target region
  // to the block arguments in front of their parent and we place them into
  // the map operands list for consistency.
  //
  // These indirect uses (via accesses to their parent) will still be
  // mapped individually in most cases, and a parent mapping doesn't
  // guarantee the parent will be mapped in its totality, partial
  // mapping is common.
  //
  // For example:
  //    map(tofrom: x%y)
  //
  // Will generate a mapping for "x" (the parent) and "y" (the member).
  // The parent "x" will not be mapped, but the member "y" will.
  // However, we must have the parent as a BlockArg and MapOperand
  // in these cases, to maintain the correct uses within the region and
  // to help tracking that the member is part of a larger object.
  //
  // In the case of:
  //    map(tofrom: x%y, x%z)
  //
  // The parent member becomes more critical, as we perform a partial
  // structure mapping where we link the mapping of the members y
  // and z together via the parent x. We do this at a kernel argument
  // level in LLVM IR and not just MLIR, which is important to maintain
  // similarity to Clang and for the runtime to do the correct thing.
  // However, we still do not map the structure in its totality but
  // rather we generate an un-sized "binding" map entry for it.
  //
  // In the case of:
  //    map(tofrom: x, x%y, x%z)
  //
  // We do actually map the entirety of "x", so the explicit mapping of
  // x%y, x%z becomes unnecessary. It is redundant to write this from a
  // Fortran OpenMP perspective (although it is legal), as even if the
  // members were allocatables or pointers, we are mandated by the
  // specification to map these (and any recursive components) in their
  // entirety, which is different to the C++ equivalent, which requires
  // explicit mapping of these segments.
  void addImplicitMembersToTarget(mlir::omp::MapInfoOp op,
                                  fir::FirOpBuilder &builder,
                                  mlir::Operation *target) {
    auto mapClauseOwner =
        llvm::dyn_cast_if_present<mlir::omp::MapClauseOwningOpInterface>(
            target);
    // TargetDataOp is technically a MapClauseOwningOpInterface, so we
    // do not need to explicitly check for the extra cases here for use_device
    // addr/ptr
    if (!mapClauseOwner)
      return;

    auto addOperands = [&](mlir::MutableOperandRange &mutableOpRange,
                           mlir::Operation *directiveOp,
                           unsigned blockArgInsertIndex = 0) {
      if (!llvm::is_contained(mutableOpRange.getAsOperandRange(),
                              op.getResult()))
        return;

      // There doesn't appear to be a simple way to convert MutableOperandRange
      // to a vector currently, so we instead use a for_each to populate our
      // vector.
      llvm::SmallVector<mlir::Value> newMapOps;
      newMapOps.reserve(mutableOpRange.size());
      llvm::for_each(
          mutableOpRange.getAsOperandRange(),
          [&newMapOps](mlir::Value oper) { newMapOps.push_back(oper); });

      for (auto mapMember : op.getMembers()) {
        if (llvm::is_contained(mutableOpRange.getAsOperandRange(), mapMember))
          continue;
        newMapOps.push_back(mapMember);
        if (directiveOp) {
          directiveOp->getRegion(0).insertArgument(
              blockArgInsertIndex, mapMember.getType(), mapMember.getLoc());
          blockArgInsertIndex++;
        }
      }

      mutableOpRange.assign(newMapOps);
    };

    auto argIface =
        llvm::dyn_cast<mlir::omp::BlockArgOpenMPOpInterface>(target);

    if (auto mapClauseOwner =
            llvm::dyn_cast<mlir::omp::MapClauseOwningOpInterface>(target)) {
      mlir::MutableOperandRange mapMutableOpRange =
          mapClauseOwner.getMapVarsMutable();
      unsigned blockArgInsertIndex =
          argIface
              ? argIface.getMapBlockArgsStart() + argIface.numMapBlockArgs()
              : 0;
      addOperands(mapMutableOpRange,
                  llvm::dyn_cast_if_present<mlir::omp::TargetOp>(
                      argIface.getOperation()),
                  blockArgInsertIndex);
    }

    if (auto targetDataOp = llvm::dyn_cast<mlir::omp::TargetDataOp>(target)) {
      mlir::MutableOperandRange useDevAddrMutableOpRange =
          targetDataOp.getUseDeviceAddrVarsMutable();
      addOperands(useDevAddrMutableOpRange, target,
                  argIface.getUseDeviceAddrBlockArgsStart() +
                      argIface.numUseDeviceAddrBlockArgs());

      mlir::MutableOperandRange useDevPtrMutableOpRange =
          targetDataOp.getUseDevicePtrVarsMutable();
      addOperands(useDevPtrMutableOpRange, target,
                  argIface.getUseDevicePtrBlockArgsStart() +
                      argIface.numUseDevicePtrBlockArgs());
    }
  }

  // We retrieve the first user that is a Target operation, of which
  // there should only be one currently. Every MapInfoOp can be tied to
  // at most one Target operation and at the minimum no operations.
  // This may change in the future with IR cleanups/modifications,
  // in which case this pass will need updating to support cases
  // where a map can have more than one user and more than one of
  // those users can be a Target operation. For now, we simply
  // return the first target operation encountered, which may
  // be on the parent MapInfoOp in the case of a member mapping.
  // In that case, we traverse the MapInfoOp chain until we
  // find the first TargetOp user.
  mlir::Operation *getFirstTargetUser(mlir::omp::MapInfoOp mapOp) {
    for (auto *user : mapOp->getUsers()) {
      if (llvm::isa<mlir::omp::TargetOp, mlir::omp::TargetDataOp,
                    mlir::omp::TargetUpdateOp, mlir::omp::TargetExitDataOp,
                    mlir::omp::TargetEnterDataOp,
                    mlir::omp::DeclareMapperInfoOp>(user))
        return user;

      if (auto mapUser = llvm::dyn_cast<mlir::omp::MapInfoOp>(user))
        return getFirstTargetUser(mapUser);
    }

    return nullptr;
  }

  // This pass executes on omp::MapInfoOp's containing descriptor based types
  // (allocatables, pointers, assumed shape etc.) and expanding them into
  // multiple omp::MapInfoOp's for each pointer member contained within the
  // descriptor.
  //
  // From the perspective of the MLIR pass manager this runs on the top level
  // operation (usually function) containing the MapInfoOp because this pass
  // will mutate siblings of MapInfoOp.
  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    if (!module)
      module = getOperation()->getParentOfType<mlir::ModuleOp>();
    fir::KindMapping kindMap = fir::getKindMapping(module);
    fir::FirOpBuilder builder{module, std::move(kindMap)};

    // We wish to maintain some function level scope (currently
    // just local function scope variables used to load and store box
    // variables into so we can access their base address, an
    // quirk of box_offset requires us to have an in memory box, but Fortran
    // in certain cases does not provide this) whilst not subjecting
    // ourselves to the possibility of race conditions while this pass
    // undergoes frequent re-iteration for the near future. So we loop
    // over function in the module and then map.info inside of those.
    getOperation()->walk([&](mlir::Operation *func) {
      if (!(mlir::isa<mlir::func::FuncOp>(func) ||
            mlir::isa<mlir::omp::DeclareMapperOp>(func)))
        return;
      // clear all local allocations we made for any boxes in any prior
      // iterations from previous function scopes.
      localBoxAllocas.clear();

      // First, walk `omp.map.info` ops to see if any record members should be
      // implicitly mapped.
      func->walk([&](mlir::omp::MapInfoOp op) {
        mlir::Type underlyingType =
            fir::unwrapRefType(op.getVarPtr().getType());

        // TODO Test with and support more complicated cases; like arrays for
        // records, for example.
        if (!fir::isRecordWithAllocatableMember(underlyingType))
          return mlir::WalkResult::advance();

        // TODO For now, only consider `omp.target` ops. Other ops that support
        // `map` clauses will follow later.
        mlir::omp::TargetOp target =
            mlir::dyn_cast_if_present<mlir::omp::TargetOp>(
                getFirstTargetUser(op));

        if (!target)
          return mlir::WalkResult::advance();

        auto mapClauseOwner =
            llvm::dyn_cast<mlir::omp::MapClauseOwningOpInterface>(*target);

        int64_t mapVarIdx = mapClauseOwner.getOperandIndexForMap(op);
        assert(mapVarIdx >= 0 &&
               mapVarIdx <
                   static_cast<int64_t>(mapClauseOwner.getMapVars().size()));

        auto argIface =
            llvm::dyn_cast<mlir::omp::BlockArgOpenMPOpInterface>(*target);
        // TODO How should `map` block argument that correspond to: `private`,
        // `use_device_addr`, `use_device_ptr`, be handled?
        mlir::BlockArgument opBlockArg = argIface.getMapBlockArgs()[mapVarIdx];
        llvm::SetVector<mlir::Operation *> mapVarForwardSlice;
        mlir::getForwardSlice(opBlockArg, &mapVarForwardSlice);

        mapVarForwardSlice.remove_if([&](mlir::Operation *sliceOp) {
          // TODO Support coordinate_of ops.
          //
          // TODO Support call ops by recursively examining the forward slice of
          // the corresponding parameter to the field in the called function.
          return !mlir::isa<hlfir::DesignateOp>(sliceOp);
        });

        auto recordType = mlir::cast<fir::RecordType>(underlyingType);
        llvm::SmallVector<mlir::Value> newMapOpsForFields;
        llvm::SmallVector<int64_t> fieldIndicies;

        for (auto fieldMemTyPair : recordType.getTypeList()) {
          auto &field = fieldMemTyPair.first;
          auto memTy = fieldMemTyPair.second;

          bool shouldMapField =
              llvm::find_if(mapVarForwardSlice, [&](mlir::Operation *sliceOp) {
                if (!fir::isAllocatableType(memTy))
                  return false;

                auto designateOp = mlir::dyn_cast<hlfir::DesignateOp>(sliceOp);
                if (!designateOp)
                  return false;

                return designateOp.getComponent() &&
                       designateOp.getComponent()->strref() == field;
              }) != mapVarForwardSlice.end();

          // TODO Handle recursive record types. Adapting
          // `createParentSymAndGenIntermediateMaps` to work direclty on MLIR
          // entities might be helpful here.

          if (!shouldMapField)
            continue;

          int64_t fieldIdx = recordType.getFieldIndex(field);
          bool alreadyMapped = [&]() {
            if (op.getMembersIndexAttr())
              for (auto indexList : op.getMembersIndexAttr()) {
                auto indexListAttr = mlir::cast<mlir::ArrayAttr>(indexList);
                if (indexListAttr.size() == 1 &&
                    mlir::cast<mlir::IntegerAttr>(indexListAttr[0]).getInt() ==
                        fieldIdx)
                  return true;
              }

            return false;
          }();

          if (alreadyMapped)
            continue;

          builder.setInsertionPoint(op);
          mlir::Value fieldIdxVal = builder.createIntegerConstant(
              op.getLoc(), mlir::IndexType::get(builder.getContext()),
              fieldIdx);
          auto fieldCoord = builder.create<fir::CoordinateOp>(
              op.getLoc(), builder.getRefType(memTy), op.getVarPtr(),
              fieldIdxVal);
          fir::factory::AddrAndBoundsInfo info =
              fir::factory::getDataOperandBaseAddr(
                  builder, fieldCoord, /*isOptional=*/false, op.getLoc());
          llvm::SmallVector<mlir::Value> bounds =
              fir::factory::genImplicitBoundsOps<mlir::omp::MapBoundsOp,
                                                 mlir::omp::MapBoundsType>(
                  builder, info,
                  hlfir::translateToExtendedValue(op.getLoc(), builder,
                                                  hlfir::Entity{fieldCoord})
                      .first,
                  /*dataExvIsAssumedSize=*/false, op.getLoc());

          mlir::omp::MapInfoOp fieldMapOp =
              builder.create<mlir::omp::MapInfoOp>(
                  op.getLoc(), fieldCoord.getResult().getType(),
                  fieldCoord.getResult(),
                  mlir::TypeAttr::get(
                      fir::unwrapRefType(fieldCoord.getResult().getType())),
                  /*varPtrPtr=*/mlir::Value{},
                  /*members=*/mlir::ValueRange{},
                  /*members_index=*/mlir::ArrayAttr{},
                  /*bounds=*/bounds, op.getMapTypeAttr(),
                  builder.getAttr<mlir::omp::VariableCaptureKindAttr>(
                      mlir::omp::VariableCaptureKind::ByRef),
                  builder.getStringAttr(op.getNameAttr().strref() + "." +
                                        field + ".implicit_map"),
                  /*partial_map=*/builder.getBoolAttr(false));
          newMapOpsForFields.emplace_back(fieldMapOp);
          fieldIndicies.emplace_back(fieldIdx);
        }

        if (newMapOpsForFields.empty())
          return mlir::WalkResult::advance();

        op.getMembersMutable().append(newMapOpsForFields);
        llvm::SmallVector<llvm::SmallVector<int64_t>> newMemberIndices;
        mlir::ArrayAttr oldMembersIdxAttr = op.getMembersIndexAttr();

        if (oldMembersIdxAttr)
          for (mlir::Attribute indexList : oldMembersIdxAttr) {
            llvm::SmallVector<int64_t> listVec;

            for (mlir::Attribute index : mlir::cast<mlir::ArrayAttr>(indexList))
              listVec.push_back(mlir::cast<mlir::IntegerAttr>(index).getInt());

            newMemberIndices.emplace_back(std::move(listVec));
          }

        for (int64_t newFieldIdx : fieldIndicies)
          newMemberIndices.emplace_back(
              llvm::SmallVector<int64_t>(1, newFieldIdx));

        op.setMembersIndexAttr(builder.create2DI64ArrayAttr(newMemberIndices));
        op.setPartialMap(true);

        return mlir::WalkResult::advance();
      });

      func->walk([&](mlir::omp::MapInfoOp op) {
        // TODO: Currently only supports a single user for the MapInfoOp. This
        // is fine for the moment, as the Fortran frontend will generate a
        // new MapInfoOp with at most one user currently. In the case of
        // members of other objects, like derived types, the user would be the
        // parent. In cases where it's a regular non-member map, the user would
        // be the target operation it is being mapped by.
        //
        // However, when/if we optimise/cleanup the IR we will have to extend
        // this pass to support multiple users, as we may wish to have a map
        // be re-used by multiple users (e.g. across multiple targets that map
        // the variable and have identical map properties).
        assert(llvm::hasSingleElement(op->getUsers()) &&
               "OMPMapInfoFinalization currently only supports single users "
               "of a MapInfoOp");

        if (fir::isTypeWithDescriptor(op.getVarType()) ||
            mlir::isa_and_present<fir::BoxAddrOp>(
                op.getVarPtr().getDefiningOp())) {
          builder.setInsertionPoint(op);
          mlir::Operation *targetUser = getFirstTargetUser(op);
          assert(targetUser && "expected user of map operation was not found");
          genDescriptorMemberMaps(op, builder, targetUser);
        }
      });

      // Wait until after we have generated all of our maps to add them onto
      // the target's block arguments, simplifying the process as there would be
      // no need to avoid accidental duplicate additions.
      func->walk([&](mlir::omp::MapInfoOp op) {
        mlir::Operation *targetUser = getFirstTargetUser(op);
        assert(targetUser && "expected user of map operation was not found");
        addImplicitMembersToTarget(op, builder, targetUser);
      });
    });
  }
};

} // namespace

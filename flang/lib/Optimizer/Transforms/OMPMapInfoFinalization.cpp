//===- OMPMapInfoFinalization.cpp
//---------------------------------------------------===//
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

#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Dialect/Support/KindMapping.h"
#include "flang/Optimizer/Transforms/Passes.h"
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

namespace fir {
#define GEN_PASS_DEF_OMPMAPINFOFINALIZATIONPASS
#include "flang/Optimizer/Transforms/Passes.h.inc"
} // namespace fir

namespace {
class OMPMapInfoFinalizationPass
    : public fir::impl::OMPMapInfoFinalizationPassBase<
          OMPMapInfoFinalizationPass> {
  /// Helper class tracking a members parent and its
  /// placement in the parents member list
  struct ParentAndPlacement {
    mlir::omp::MapInfoOp parent;
    size_t index;
  };

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
    for (auto *users : op->getUsers())
      if (auto map = mlir::dyn_cast_if_present<mlir::omp::MapInfoOp>(users))
        for (auto [i, mapMember] : llvm::enumerate(map.getMembers()))
          if (mapMember.getDefiningOp() == op)
            mapMemberUsers.push_back({map, i});
  }

  /// Returns the integer numbers contained within the mlir::Attributes within
  /// the values array.
  llvm::SmallVector<int64_t>
  getAsIntegers(llvm::ArrayRef<mlir::Attribute> values) {
    llvm::SmallVector<int64_t> ints;
    ints.reserve(values.size());
    llvm::transform(values, std::back_inserter(ints),
                    [](mlir::Attribute value) {
                      return mlir::cast<mlir::IntegerAttr>(value).getInt();
                    });
    return ints;
  }

  /// This function will expand a MapInfoOp's member indices back into a vector
  /// so that they can be trivially modified as unfortunately the attribute type
  /// that's used does not have modifiable fields at the moment (generally
  /// awkward to work with)
  void getMemberIndicesAsVectors(
      mlir::omp::MapInfoOp mapInfo,
      llvm::SmallVector<llvm::SmallVector<int64_t>> &indices) {
    indices.reserve(mapInfo.getMembersIndexAttr().getValue().size());
    for (auto v : mapInfo.getMembersIndexAttr().getValue()) {
      auto memberIndex = mlir::cast<mlir::ArrayAttr>(v);
      indices.push_back(getAsIntegers(memberIndex.getValue()));
    }
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

    // The fir::BoxOffsetOp only works with !fir.ref<!fir.box<...>> types, as
    // allowing it to access non-reference box operations can cause some
    // problematic SSA IR. However, in the case of assumed shape's the type
    // is not a !fir.ref, in these cases to retrieve the appropriate
    // !fir.ref<!fir.box<...>> to access the data we need to map we must
    // perform an alloca and then store to it and retrieve the data from the new
    // alloca.
    if (mlir::isa<fir::BaseBoxType>(descriptor.getType())) {
      mlir::OpBuilder::InsertPoint insPt = builder.saveInsertionPoint();
      mlir::Block *allocaBlock = builder.getAllocaBlock();
      mlir::Location loc = boxMap->getLoc();
      assert(allocaBlock && "No alloca block found for this top level op");
      builder.setInsertionPointToStart(allocaBlock);
      auto alloca = builder.create<fir::AllocaOp>(loc, descriptor.getType());
      builder.restoreInsertionPoint(insPt);
      builder.create<fir::StoreOp>(loc, descriptor, alloca);
      descriptor = alloca;
    }

    return descriptor;
  }

  /// Simple function that will generate a FIR operation accessing
  /// the descriptors base address (BoxOffsetOp) and then generate a
  /// MapInfoOp for it, the most important thing to note is that
  /// we normally move the bounds from the descriptor map onto the
  /// base address map.
  mlir::omp::MapInfoOp getBaseAddrMap(mlir::Value descriptor,
                                      mlir::OperandRange bounds,
                                      int64_t mapType,
                                      fir::FirOpBuilder &builder) {
    mlir::Location loc = descriptor.getLoc();
    mlir::Value baseAddrAddr = builder.create<fir::BoxOffsetOp>(
        loc, descriptor, fir::BoxFieldAttr::base_addr);

    // Member of the descriptor pointing at the allocated data
    return builder.create<mlir::omp::MapInfoOp>(
        loc, baseAddrAddr.getType(), descriptor,
        mlir::TypeAttr::get(llvm::cast<mlir::omp::PointerLikeType>(
                                fir::unwrapRefType(baseAddrAddr.getType()))
                                .getElementType()),
        baseAddrAddr, /*members=*/mlir::SmallVector<mlir::Value>{},
        /*membersIndex=*/mlir::ArrayAttr{}, bounds,
        builder.getIntegerAttr(builder.getIntegerType(64, false), mapType),
        builder.getAttr<mlir::omp::VariableCaptureKindAttr>(
            mlir::omp::VariableCaptureKind::ByRef),
        /*name=*/builder.getStringAttr(""),
        /*partial_map=*/builder.getBoolAttr(false));
  }

  /// This function adjusts the member indices vector to include a new
  /// base address member, we take the position of the descriptor in
  /// the member indices list, which is the index data that the base
  /// addresses index will be based off of, as the base address is
  /// a member of the descriptor, we must also alter other members
  /// indices in the list to account for this new addition. This
  /// requires inserting into the middle of a member index vector
  /// in some cases (i.e. we could be accessing the member of a
  /// descriptor type with a subsequent map, so we must be sure to
  /// adjust any of these cases with the addition of the new base
  /// address index value).
  void adjustMemberIndices(
      llvm::SmallVector<llvm::SmallVector<int64_t>> &memberIndices,
      size_t memberIndex) {
    llvm::SmallVector<int64_t> baseAddrIndex = memberIndices[memberIndex];
    baseAddrIndex.push_back(0);

    // If we find another member that is "derived/a member of" the descriptor
    // that is not the descriptor itself, we must insert a 0 for the new base
    // address we have just added for the descriptor into the list at the
    // appropriate position to maintain correctness of the positional/index data
    // for that member.
    size_t insertPosition =
        std::distance(baseAddrIndex.begin(), std::prev(baseAddrIndex.end()));
    for (size_t i = 0; i < memberIndices.size(); ++i) {
      if (memberIndices[i].size() > insertPosition &&
          std::equal(baseAddrIndex.begin(), std::prev(baseAddrIndex.end()),
                     memberIndices[i].begin())) {
        memberIndices[i].insert(
            std::next(memberIndices[i].begin(), insertPosition), 0);
      }
    }

    // Insert our newly created baseAddrIndex into the larger list of indices at
    // the correct location.
    memberIndices.insert(std::next(memberIndices.begin(), memberIndex + 1),
                         baseAddrIndex);
  }

  /// Adjusts the descriptors map type the main alteration that is done
  /// currently is transforming the map type to OMP_MAP_TO where possible.
  // This is because we will always need to map the descriptor to device
  /// (or at the very least it seems to be the case currently with the
  /// current lowered kernel IR), as without the appropriate descriptor
  /// information on the device there is a risk of the kernel IR
  /// requesting for various data that will not have been copied to
  /// perform things like indexing, this can cause segfaults and
  /// memory access errors. However, we do not need this data mapped
  /// back to the host from the device, as we cannot alter the data
  /// via resizing or deletion on the device, this is specified in the
  /// OpenMP specification, so discarding any descriptor alterations via
  /// no map back is reasonable (and required for certain segments
  /// of descriptor data like the type descriptor that are global
  /// constants). This alteration is only unapplicable to
  /// target exit and target update currently, and that's due to
  /// target exit not allowing To mappings, and target update not
  /// allowing both to and from simultaneously. We currently try
  /// to maintain the implicit flag where neccesary, although, it
  /// does not seem strictly required.
  unsigned long getDescriptorMapType(unsigned long mapTypeFlag,
                                     mlir::Operation *target) {
    if (llvm::isa_and_nonnull<mlir::omp::TargetExitDataOp>(target) ||
        llvm::isa_and_nonnull<mlir::omp::TargetUpdateOp>(target))
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
    auto baseAddr = getBaseAddrMap(descriptor, op.getBounds(),
                                   op.getMapType().value_or(0), builder);
    mlir::ArrayAttr newMembersAttr;
    mlir::SmallVector<mlir::Value> newMembers;
    llvm::SmallVector<llvm::SmallVector<int64_t>> memberIndices;

    if (!mapMemberUsers.empty() || !op.getMembers().empty())
      getMemberIndicesAsVectors(
          !mapMemberUsers.empty() ? mapMemberUsers[0].parent : op,
          memberIndices);

    // If the operation that we are expanding with a descriptor has a user
    // (parent), then we have to expand the parents member indices to reflect
    // the adjusted member indices for the base address insertion. However, if
    // it does not then we are expanding a MapInfoOp without any pre-existing
    // member information to now have one new member for the base address or we
    // are expanding a parent that is a descriptor and we have to adjust all of
    // it's members to reflect the insertion of the base address.
    if (!mapMemberUsers.empty()) {
      // Currently, there should only be one user per map when this pass
      // is executed, either a parent map, holding the current map in its
      // member list, or a target operation that holds a map clause. This
      // may change in the future if we aim to refactor the MLIR for map
      // clauses to allow sharing of duplicate maps across target
      // operations.
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
          builder.create2DIntegerArrayAttr(memberIndices));
    } else {
      newMembers.push_back(baseAddr);
      if (!op.getMembers().empty()) {
        for (auto &indices : memberIndices)
          indices.insert(indices.begin(), 0);
        memberIndices.insert(memberIndices.begin(), {0});
        newMembersAttr = builder.create2DIntegerArrayAttr(memberIndices);
        newMembers.append(op.getMembers().begin(), op.getMembers().end());
      } else {
        llvm::SmallVector<llvm::SmallVector<int64_t>> memberIdx = {{0}};
        newMembersAttr = builder.create2DIntegerArrayAttr(memberIdx);
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
    if (!mapClauseOwner)
      return;

    llvm::SmallVector<mlir::Value> newMapOps;
    mlir::OperandRange mapVarsArr = mapClauseOwner.getMapVars();
    auto targetOp = llvm::dyn_cast<mlir::omp::TargetOp>(target);

    for (size_t i = 0; i < mapVarsArr.size(); ++i) {
      if (mapVarsArr[i] == op) {
        for (auto [j, mapMember] : llvm::enumerate(op.getMembers())) {
          newMapOps.push_back(mapMember);
          // for TargetOp's which have IsolatedFromAbove we must align the
          // new additional map operand with an appropriate BlockArgument,
          // as the printing and later processing currently requires a 1:1
          // mapping of BlockArgs to MapInfoOp's at the same placement in
          // each array (BlockArgs and MapVars).
          if (targetOp) {
            targetOp.getRegion().insertArgument(i + j, mapMember.getType(),
                                                targetOp->getLoc());
          }
        }
      }
      newMapOps.push_back(mapVarsArr[i]);
    }
    mapClauseOwner.getMapVarsMutable().assign(newMapOps);
  }

  // We retrieve the first user that is a Target operation, there
  // should only be one currently, every MapInfoOp can be tied to
  // at most 1 Target operation and at the minimum no operation,
  // this may change in the future with IR cleanups/modifications
  // in which case this pass will need updated to support cases
  // where a map can have more than one user and more than one of
  // those users can be a Target operation. For now, we simply
  // return the first target operation encountered, which may
  // be on the parent MapInfoOp in the case of a member mapping
  // in which case we must traverse the MapInfoOp chain until we
  // find the first TargetOp user.
  mlir::Operation *getFirstTargetUser(mlir::omp::MapInfoOp mapOp) {
    for (auto *user : mapOp->getUsers()) {
      if (llvm::isa<mlir::omp::TargetOp, mlir::omp::TargetDataOp,
                    mlir::omp::TargetUpdateOp, mlir::omp::TargetExitDataOp,
                    mlir::omp::TargetEnterDataOp>(user))
        return user;

      if (auto mapUser = llvm::dyn_cast_if_present<mlir::omp::MapInfoOp>(user))
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
    mlir::ModuleOp module =
        mlir::dyn_cast_or_null<mlir::ModuleOp>(getOperation());
    if (!module)
      module = getOperation()->getParentOfType<mlir::ModuleOp>();
    fir::KindMapping kindMap = fir::getKindMapping(module);
    fir::FirOpBuilder builder{module, std::move(kindMap)};

    getOperation()->walk([&](mlir::omp::MapInfoOp op) {
      // TODO: Currently only supports a single user for the MapInfoOp, this
      // is fine for the moment as the Fortran frontend will generate a
      // new MapInfoOp with at most one user currently, in the case of
      // members of other objects like derived types, the user would be the
      // parent, in cases where it's a regular non-member map the user would
      // be the target operation it is being mapped by.
      //
      // However, when/if we optimise/cleanup the IR we will have to extend
      // this pass to support multiple users, as I would imagine we may wish
      // to have a map be re-used by multiple users (e.g. across multiple
      // targets that map the variable and have identical map properties)
      assert(llvm::hasSingleElement(op->getUsers()) &&
             "OMPMapInfoFinalization currently only supports single users "
             "of a MapInfoOp");

      if (fir::isTypeWithDescriptor(op.getVarType()) ||
          mlir::isa_and_present<fir::BoxAddrOp>(
              op.getVarPtr().getDefiningOp())) {
        builder.setInsertionPoint(op);
        genDescriptorMemberMaps(op, builder, getFirstTargetUser(op));
      }
    });

    // Wait until after we have generated all of our maps to add them onto
    // the targets block arguments, simplifying the process as no need to
    // avoid accidental duplicate additions
    getOperation()->walk([&](mlir::omp::MapInfoOp op) {
      addImplicitMembersToTarget(op, builder, getFirstTargetUser(op));
    });
  }
};

} // namespace

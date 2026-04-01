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
#include "aiir/Analysis/SliceAnalysis.h"
#include "aiir/Dialect/Func/IR/FuncOps.h"
#include "aiir/Dialect/OpenMP/OpenMPDialect.h"
#include "aiir/IR/BuiltinDialect.h"
#include "aiir/IR/BuiltinOps.h"
#include "aiir/IR/Operation.h"
#include "aiir/IR/SymbolTable.h"
#include "aiir/Pass/Pass.h"
#include "aiir/Support/LLVM.h"
#include "llvm/ADT/BitmaskEnum.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cstddef>
#include <iterator>
#include <numeric>

#define DEBUG_TYPE "omp-map-info-finalization"

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
    aiir::omp::MapInfoOp parent;
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
  std::map<aiir::Operation *, aiir::Value> localBoxAllocas;

  // List of deferrable descriptors to process at the end of
  // the pass.
  llvm::SmallVector<aiir::Operation *> deferrableDesc;

  /// Return true if the given path exists in a list of paths.
  static bool
  containsPath(const llvm::SmallVectorImpl<llvm::SmallVector<int64_t>> &paths,
               llvm::ArrayRef<int64_t> path) {
    return llvm::any_of(paths, [&](const llvm::SmallVector<int64_t> &p) {
      return p.size() == path.size() &&
             std::equal(p.begin(), p.end(), path.begin());
    });
  }

  /// Return true if the given path is already present in
  /// op.getMembersIndexAttr().
  static bool mappedIndexPathExists(aiir::omp::MapInfoOp op,
                                    llvm::ArrayRef<int64_t> indexPath) {
    if (aiir::ArrayAttr attr = op.getMembersIndexAttr()) {
      for (aiir::Attribute list : attr) {
        auto listAttr = aiir::cast<aiir::ArrayAttr>(list);
        if (listAttr.size() != indexPath.size())
          continue;
        bool allEq = true;
        for (auto [i, val] : llvm::enumerate(listAttr)) {
          if (aiir::cast<aiir::IntegerAttr>(val).getInt() != indexPath[i]) {
            allEq = false;
            break;
          }
        }
        if (allEq)
          return true;
      }
    }
    return false;
  }

  /// Build a compact string key for an index path for set-based
  /// deduplication. Format: "N:v0,v1,..." where N is the length.
  static void buildPathKey(llvm::ArrayRef<int64_t> path,
                           llvm::SmallString<64> &outKey) {
    outKey.clear();
    llvm::raw_svector_ostream os(outKey);
    os << path.size() << ':';
    for (size_t i = 0; i < path.size(); ++i) {
      if (i)
        os << ',';
      os << path[i];
    }
  }

  /// Return true if the module has an OpenMP requires clause that includes
  /// unified_shared_memory.
  static bool moduleRequiresUSM(aiir::ModuleOp module) {
    assert(module && "invalid module");
    if (auto req = module->getAttrOfType<aiir::omp::ClauseRequiresAttr>(
            "omp.requires"))
      return aiir::omp::bitEnumContainsAll(
          req.getValue(), aiir::omp::ClauseRequires::unified_shared_memory);
    return false;
  }

  /// Create the member map for coordRef and append it (and its index
  /// path) to the provided new* vectors, if it is not already present.
  void appendMemberMapIfNew(
      aiir::omp::MapInfoOp op, fir::FirOpBuilder &builder, aiir::Location loc,
      aiir::Value coordRef, llvm::ArrayRef<int64_t> indexPath,
      llvm::StringRef memberName,
      llvm::SmallVectorImpl<aiir::Value> &newMapOpsForFields,
      llvm::SmallVectorImpl<llvm::SmallVector<int64_t>> &newMemberIndexPaths) {
    // Local de-dup within this op invocation.
    if (containsPath(newMemberIndexPaths, indexPath))
      return;
    // Global de-dup against already present member indices.
    if (mappedIndexPathExists(op, indexPath))
      return;

    if (op.getMapperId()) {
      aiir::omp::DeclareMapperOp symbol =
          aiir::SymbolTable::lookupNearestSymbolFrom<
              aiir::omp::DeclareMapperOp>(op, op.getMapperIdAttr());
      assert(symbol && "missing symbol for declare mapper identifier");
      aiir::omp::DeclareMapperInfoOp mapperInfo = symbol.getDeclareMapperInfo();
      // TODO: Probably a way to cache these keys in someway so we don't
      // constantly go through the process of rebuilding them on every check, to
      // save some cycles, but it can wait for a subsequent patch.
      for (auto v : mapperInfo.getMapVars()) {
        aiir::omp::MapInfoOp map =
            aiir::cast<aiir::omp::MapInfoOp>(v.getDefiningOp());
        if (!map.getMembers().empty() && mappedIndexPathExists(map, indexPath))
          return;
      }
    }

    builder.setInsertionPoint(op);
    fir::factory::AddrAndBoundsInfo info = fir::factory::getDataOperandBaseAddr(
        builder, coordRef, /*isOptional=*/false, loc);
    llvm::SmallVector<aiir::Value> bounds = fir::factory::genImplicitBoundsOps<
        aiir::omp::MapBoundsOp, aiir::omp::MapBoundsType>(
        builder, info,
        hlfir::translateToExtendedValue(loc, builder, hlfir::Entity{coordRef})
            .first,
        /*dataExvIsAssumedSize=*/false, loc);

    aiir::omp::MapInfoOp fieldMapOp = aiir::omp::MapInfoOp::create(
        builder, loc, coordRef.getType(), coordRef,
        aiir::TypeAttr::get(fir::unwrapRefType(coordRef.getType())),
        op.getMapTypeAttr(),
        builder.getAttr<aiir::omp::VariableCaptureKindAttr>(
            aiir::omp::VariableCaptureKind::ByRef),
        /*varPtrPtr=*/aiir::Value{}, /*members=*/aiir::ValueRange{},
        /*members_index=*/aiir::ArrayAttr{}, bounds,
        /*mapperId=*/aiir::FlatSymbolRefAttr(),
        builder.getStringAttr(op.getNameAttr().strref() + "." + memberName +
                              ".implicit_map"),
        /*partial_map=*/builder.getBoolAttr(false));

    newMapOpsForFields.emplace_back(fieldMapOp);
    newMemberIndexPaths.emplace_back(indexPath.begin(), indexPath.end());
  }

  // Check if the declaration operation we have refers to a dummy
  // function argument.
  bool isDummyArgument(aiir::Value mappedValue) {
    if (auto declareOp = aiir::dyn_cast_if_present<hlfir::DeclareOp>(
            mappedValue.getDefiningOp()))
      if (auto dummyScope = declareOp.getDummyScope())
        return true;
    return false;
  }

  // Relevant for OpenMP < 5.2, where attach semantics and rules don't exist.
  // As descriptors were an unspoken implementation detail in these versions
  // there's certain cases where the user (and the compiler implementation)
  // can create data mapping errors by having temporary descriptors stuck
  // in memory. The main example is calling an 'target enter data map'
  // without a corresponding exit on an assumed shape or size dummy
  // argument, a local stack descriptor is generated, gets mapped and
  // is then left on device. A user doesn't realize what they've done as
  // the OpenMP specification isn't explicit on descriptor handling in
  // earlier versions and as far as Fortran is concerned this si something
  // hidden from a user. To avoid this we can defer the descriptor mapping
  // in these cases until target or target data regions, when we can be
  // sure they have a clear limited scope on device.
  bool canDeferDescriptorMapping(aiir::Value descriptor) {
    if (fir::isAllocatableType(descriptor.getType()) ||
        fir::isPointerType(descriptor.getType()))
      return false;
    if (isDummyArgument(descriptor) &&
        (fir::isAssumedType(descriptor.getType()) ||
         fir::isAssumedShape(descriptor.getType())))
      return true;
    return false;
  }

  /// getMemberUserList gathers all users of a particular MapInfoOp that are
  /// other MapInfoOp's and places them into the mapMemberUsers list, which
  /// records the map that the current argument MapInfoOp "op" is part of
  /// alongside the placement of "op" in the recorded users members list. The
  /// intent of the generated list is to find all MapInfoOp's that may be
  /// considered parents of the passed in "op" and in which it shows up in the
  /// member list, alongside collecting the placement information of "op" in its
  /// parents member list.
  void
  getMemberUserList(aiir::omp::MapInfoOp op,
                    llvm::SmallVectorImpl<ParentAndPlacement> &mapMemberUsers) {
    for (auto *user : op->getUsers())
      if (auto map = aiir::dyn_cast_if_present<aiir::omp::MapInfoOp>(user))
        for (auto [i, mapMember] : llvm::enumerate(map.getMembers()))
          if (mapMember.getDefiningOp() == op)
            mapMemberUsers.push_back({map, i});
  }

  void getAsIntegers(llvm::ArrayRef<aiir::Attribute> values,
                     llvm::SmallVectorImpl<int64_t> &ints) {
    ints.reserve(values.size());
    llvm::transform(values, std::back_inserter(ints),
                    [](aiir::Attribute value) {
                      return aiir::cast<aiir::IntegerAttr>(value).getInt();
                    });
  }

  /// This function will expand a MapInfoOp's member indices back into a vector
  /// so that they can be trivially modified as unfortunately the attribute type
  /// that's used does not have modifiable fields at the moment (generally
  /// awkward to work with)
  void getMemberIndicesAsVectors(
      aiir::omp::MapInfoOp mapInfo,
      llvm::SmallVectorImpl<llvm::SmallVector<int64_t>> &indices) {
    indices.reserve(mapInfo.getMembersIndexAttr().getValue().size());
    llvm::transform(mapInfo.getMembersIndexAttr().getValue(),
                    std::back_inserter(indices), [this](aiir::Attribute value) {
                      auto memberIndex = aiir::cast<aiir::ArrayAttr>(value);
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
  aiir::Value getDescriptorFromBoxMap(aiir::omp::MapInfoOp boxMap,
                                      fir::FirOpBuilder &builder,
                                      bool &canDescBeDeferred) {
    aiir::Value descriptor = boxMap.getVarPtr();
    if (!fir::isTypeWithDescriptor(boxMap.getVarType()))
      if (auto addrOp = aiir::dyn_cast_if_present<fir::BoxAddrOp>(
              boxMap.getVarPtr().getDefiningOp()))
        descriptor = addrOp.getVal();

    canDescBeDeferred = canDeferDescriptorMapping(descriptor);

    if (!aiir::isa<fir::BaseBoxType>(descriptor.getType()) &&
        !fir::factory::isOptionalArgument(descriptor.getDefiningOp()))
      return descriptor;

    aiir::Value &alloca = localBoxAllocas[descriptor.getDefiningOp()];
    aiir::Location loc = boxMap->getLoc();

    if (!alloca) {
      // The fir::BoxOffsetOp only works with !fir.ref<!fir.box<...>> types, as
      // allowing it to access non-reference box operations can cause some
      // problematic SSA IR. However, in the case of assumed shape's the type
      // is not a !fir.ref, in these cases to retrieve the appropriate
      // !fir.ref<!fir.box<...>> to access the data we need to map we must
      // perform an alloca and then store to it and retrieve the data from the
      // new alloca.
      aiir::OpBuilder::InsertPoint insPt = builder.saveInsertionPoint();
      aiir::Block *allocaBlock = builder.getAllocaBlock();
      assert(allocaBlock && "No alloca block found for this top level op");
      builder.setInsertionPointToStart(allocaBlock);

      aiir::Type allocaType = descriptor.getType();
      if (fir::isBoxAddress(allocaType))
        allocaType = fir::unwrapRefType(allocaType);
      alloca = fir::AllocaOp::create(builder, loc, allocaType);
      builder.restoreInsertionPoint(insPt);
    }

    // We should only emit a store if the passed in data is present, it is
    // possible a user passes in no argument to an optional parameter, in which
    // case we cannot store or we'll segfault on the emitted memcpy.
    // TODO: We currently emit a present -> load/store every time we use a
    // mapped value that requires a local allocation, this isn't the most
    // efficient, although, it is more correct in a lot of situations. One
    // such situation is emitting a this series of instructions in separate
    // segments of a branch (e.g. two target regions in separate else/if branch
    // mapping the same function argument), however, it would be nice to be able
    // to optimize these situations e.g. raising the load/store out of the
    // branch if possible. But perhaps this is best left to lower level
    // optimisation passes.
    auto isPresent =
        fir::IsPresentOp::create(builder, loc, builder.getI1Type(), descriptor);
    builder.genIfOp(loc, {}, isPresent, false)
        .genThen([&]() {
          descriptor = builder.loadIfRef(loc, descriptor);
          fir::StoreOp::create(builder, loc, descriptor, alloca);
        })
        .end();
    return alloca;
  }

  /// Function that generates a FIR operation accessing the descriptor's
  /// base address (BoxOffsetOp) and a MapInfoOp for it. The most
  /// important thing to note is that we normally move the bounds from
  /// the descriptor map onto the base address map.
  aiir::omp::MapInfoOp
  genBaseAddrMap(aiir::Value descriptor, aiir::OperandRange bounds,
                 aiir::omp::ClauseMapFlags mapType, fir::FirOpBuilder &builder,
                 aiir::FlatSymbolRefAttr mapperId = aiir::FlatSymbolRefAttr()) {
    aiir::Location loc = descriptor.getLoc();
    aiir::Value baseAddrAddr = fir::BoxOffsetOp::create(
        builder, loc, descriptor, fir::BoxFieldAttr::base_addr);

    aiir::Type underlyingVarType =
        llvm::cast<aiir::omp::PointerLikeType>(
            fir::unwrapRefType(baseAddrAddr.getType()))
            .getElementType();
    if (auto seqType = llvm::dyn_cast<fir::SequenceType>(underlyingVarType))
      if (seqType.hasDynamicExtents())
        underlyingVarType = seqType.getEleTy();

    // Member of the descriptor pointing at the allocated data
    return aiir::omp::MapInfoOp::create(
        builder, loc, baseAddrAddr.getType(), descriptor,
        aiir::TypeAttr::get(underlyingVarType),
        builder.getAttr<aiir::omp::ClauseMapFlagsAttr>(mapType),
        builder.getAttr<aiir::omp::VariableCaptureKindAttr>(
            aiir::omp::VariableCaptureKind::ByRef),
        baseAddrAddr, /*members=*/aiir::SmallVector<aiir::Value>{},
        /*membersIndex=*/aiir::ArrayAttr{}, bounds,
        /*mapperId=*/mapperId,
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
  aiir::omp::ClauseMapFlags
  getDescriptorMapType(aiir::omp::ClauseMapFlags mapTypeFlag,
                       aiir::Operation *target) {
    using mapFlags = aiir::omp::ClauseMapFlags;
    if (llvm::isa_and_nonnull<aiir::omp::TargetExitDataOp,
                              aiir::omp::TargetUpdateOp>(target))
      return mapTypeFlag;

    mapFlags flags =
        mapFlags::to | (mapTypeFlag & (mapFlags::implicit | mapFlags::always));

    // Descriptors for objects will always be copied. This is because the
    // descriptor can be rematerialized by the compiler, and so the address
    // of the descriptor for a given object at one place in the code may
    // differ from that address in another place. The contents of the
    // descriptor (the base address in particular) will remain unchanged
    // though.
    // TODO/FIXME: We currently cannot have MAP_CLOSE and MAP_ALWAYS on
    // the descriptor at once, these are mutually exclusive and when
    // both are applied the runtime will fail to map.
    flags |= ((mapFlags(mapTypeFlag) & mapFlags::close) == mapFlags::close)
                 ? mapFlags::close
                 : mapFlags::always;

    // For unified_shared_memory, we additionally add `CLOSE` on the descriptor
    // to ensure device-local placement where required by tests relying on USM +
    // close semantics.
    if (moduleRequiresUSM(target->getParentOfType<aiir::ModuleOp>()))
      flags |= mapFlags::close;
    return flags;
  }

  /// Check if the mapOp is present in the HasDeviceAddr clause on
  /// the userOp. Only applies to TargetOp.
  bool isHasDeviceAddr(aiir::omp::MapInfoOp mapOp, aiir::Operation &userOp) {
    if (auto targetOp = llvm::dyn_cast<aiir::omp::TargetOp>(userOp)) {
      for (aiir::Value hda : targetOp.getHasDeviceAddrVars()) {
        if (hda.getDefiningOp() == mapOp)
          return true;
      }
    }
    return false;
  }

  bool isUseDeviceAddr(aiir::omp::MapInfoOp mapOp, aiir::Operation &userOp) {
    if (auto targetDataOp = llvm::dyn_cast<aiir::omp::TargetDataOp>(userOp)) {
      for (aiir::Value uda : targetDataOp.getUseDeviceAddrVars()) {
        if (uda.getDefiningOp() == mapOp)
          return true;
      }
    }
    return false;
  }

  bool isUseDevicePtr(aiir::omp::MapInfoOp mapOp, aiir::Operation &userOp) {
    if (auto targetDataOp = llvm::dyn_cast<aiir::omp::TargetDataOp>(userOp)) {
      for (aiir::Value udp : targetDataOp.getUseDevicePtrVars()) {
        if (udp.getDefiningOp() == mapOp)
          return true;
      }
    }
    return false;
  }

  // Expand mappings of type(C_PTR) to map their `__address` field explicitly
  // as a single pointer-sized member (USM-gated at callsite). This helps in
  // USM scenarios to ensure the pointer-sized mapping is used.
  aiir::omp::MapInfoOp genCptrMemberMap(aiir::omp::MapInfoOp op,
                                        fir::FirOpBuilder &builder) {
    if (!op.getMembers().empty())
      return op;

    aiir::Type varTy = fir::unwrapRefType(op.getVarPtr().getType());
    if (!aiir::isa<fir::RecordType>(varTy))
      return op;
    auto recTy = aiir::cast<fir::RecordType>(varTy);
    // If not a builtin C_PTR record, skip.
    if (!recTy.getName().ends_with("__builtin_c_ptr"))
      return op;

    // Find the index of the c_ptr address component named "__address".
    int32_t fieldIdx = recTy.getFieldIndex("__address");
    if (fieldIdx < 0)
      return op;

    aiir::Location loc = op.getVarPtr().getLoc();
    aiir::Type memTy = recTy.getType(fieldIdx);
    fir::IntOrValue idxConst =
        aiir::IntegerAttr::get(builder.getI32Type(), fieldIdx);
    aiir::Value coord = fir::CoordinateOp::create(
        builder, loc, builder.getRefType(memTy), op.getVarPtr(),
        llvm::SmallVector<fir::IntOrValue, 1>{idxConst});

    // Child for the `__address` member.
    llvm::SmallVector<llvm::SmallVector<int64_t>> memberIdx = {{0}};
    aiir::ArrayAttr newMembersAttr = builder.create2DI64ArrayAttr(memberIdx);
    // Force CLOSE in USM paths so the pointer gets device-local placement
    // when required by tests relying on USM + close semantics.
    aiir::omp::ClauseMapFlagsAttr mapTypeAttr =
        builder.getAttr<aiir::omp::ClauseMapFlagsAttr>(
            op.getMapType() | aiir::omp::ClauseMapFlags::close);

    aiir::omp::MapInfoOp memberMap = aiir::omp::MapInfoOp::create(
        builder, loc, coord.getType(), coord,
        aiir::TypeAttr::get(fir::unwrapRefType(coord.getType())), mapTypeAttr,
        builder.getAttr<aiir::omp::VariableCaptureKindAttr>(
            aiir::omp::VariableCaptureKind::ByRef),
        /*varPtrPtr=*/aiir::Value{},
        /*members=*/llvm::SmallVector<aiir::Value>{},
        /*member_index=*/aiir::ArrayAttr{},
        /*bounds=*/op.getBounds(),
        /*mapperId=*/aiir::FlatSymbolRefAttr(),
        /*name=*/op.getNameAttr(),
        /*partial_map=*/builder.getBoolAttr(false));

    // Rebuild the parent as a container with the `__address` member.
    aiir::omp::MapInfoOp newParent = aiir::omp::MapInfoOp::create(
        builder, op.getLoc(), op.getResult().getType(), op.getVarPtr(),
        op.getVarTypeAttr(), mapTypeAttr, op.getMapCaptureTypeAttr(),
        /*varPtrPtr=*/aiir::Value{},
        /*members=*/llvm::SmallVector<aiir::Value>{memberMap},
        /*member_index=*/newMembersAttr,
        /*bounds=*/llvm::SmallVector<aiir::Value>{},
        /*mapperId=*/aiir::FlatSymbolRefAttr(), op.getNameAttr(),
        /*partial_map=*/builder.getBoolAttr(false));
    op.replaceAllUsesWith(newParent.getResult());
    op->erase();
    return newParent;
  }

  aiir::omp::MapInfoOp genDescriptorMemberMaps(aiir::omp::MapInfoOp op,
                                               fir::FirOpBuilder &builder,
                                               aiir::Operation *target) {
    llvm::SmallVector<ParentAndPlacement> mapMemberUsers;
    getMemberUserList(op, mapMemberUsers);

    // TODO: map the addendum segment of the descriptor, similarly to the
    // base address/data pointer member.
    bool descCanBeDeferred = false;
    aiir::Value descriptor =
        getDescriptorFromBoxMap(op, builder, descCanBeDeferred);

    aiir::ArrayAttr newMembersAttr;
    aiir::SmallVector<aiir::Value> newMembers;
    llvm::SmallVector<llvm::SmallVector<int64_t>> memberIndices;
    bool isHasDeviceAddrFlag = isHasDeviceAddr(op, *target);

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
    //
    // If we're expanding a top-level descriptor for a map operation that
    // resulted from "has_device_addr" clause, then we want the base pointer
    // from the descriptor to be used verbatim, i.e. without additional
    // remapping. To avoid this remapping, simply don't generate any map
    // information for the descriptor members.
    aiir::FlatSymbolRefAttr mapperId = op.getMapperIdAttr();
    if (!mapMemberUsers.empty()) {
      // Currently, there should only be one user per map when this pass
      // is executed. Either a parent map, holding the current map in its
      // member list, or a target operation that holds a map clause. This
      // may change in the future if we aim to refactor the AIIR for map
      // clauses to allow sharing of duplicate maps across target
      // operations.
      assert(mapMemberUsers.size() == 1 &&
             "OMPMapInfoFinalization currently only supports single users of a "
             "MapInfoOp");
      auto baseAddr = genBaseAddrMap(descriptor, op.getBounds(),
                                     op.getMapType(), builder, mapperId);
      ParentAndPlacement mapUser = mapMemberUsers[0];
      adjustMemberIndices(memberIndices, mapUser.index);
      llvm::SmallVector<aiir::Value> newMemberOps;
      for (auto v : mapUser.parent.getMembers()) {
        newMemberOps.push_back(v);
        if (v == op)
          newMemberOps.push_back(baseAddr);
      }
      mapUser.parent.getMembersMutable().assign(newMemberOps);
      mapUser.parent.setMembersIndexAttr(
          builder.create2DI64ArrayAttr(memberIndices));
    } else if (!isHasDeviceAddrFlag) {
      auto baseAddr = genBaseAddrMap(descriptor, op.getBounds(),
                                     op.getMapType(), builder, mapperId);
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

    // Descriptors for objects listed on the `has_device_addr` will always
    // be copied. This is because the descriptor can be rematerialized by the
    // compiler, and so the address of the descriptor for a given object at
    // one place in the code may differ from that address in another place.
    // The contents of the descriptor (the base address in particular) will
    // remain unchanged though.
    aiir::omp::ClauseMapFlags mapType = op.getMapType();
    if (isHasDeviceAddrFlag) {
      mapType |= aiir::omp::ClauseMapFlags::always;
    }

    aiir::omp::MapInfoOp newDescParentMapOp = aiir::omp::MapInfoOp::create(
        builder, op->getLoc(), op.getResult().getType(), descriptor,
        aiir::TypeAttr::get(fir::unwrapRefType(descriptor.getType())),
        builder.getAttr<aiir::omp::ClauseMapFlagsAttr>(
            getDescriptorMapType(mapType, target)),
        op.getMapCaptureTypeAttr(), /*varPtrPtr=*/aiir::Value{}, newMembers,
        newMembersAttr, /*bounds=*/aiir::SmallVector<aiir::Value>{},
        /*mapperId=*/aiir::FlatSymbolRefAttr(), op.getNameAttr(),
        /*partial_map=*/builder.getBoolAttr(false));
    op.replaceAllUsesWith(newDescParentMapOp.getResult());
    op->erase();

    if (descCanBeDeferred)
      deferrableDesc.push_back(newDescParentMapOp);

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
  // level in LLVM IR and not just AIIR, which is important to maintain
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
  void addImplicitMembersToTarget(aiir::omp::MapInfoOp op,
                                  fir::FirOpBuilder &builder,
                                  aiir::Operation *target) {
    auto mapClauseOwner =
        llvm::dyn_cast_if_present<aiir::omp::MapClauseOwningOpInterface>(
            target);
    // TargetDataOp is technically a MapClauseOwningOpInterface, so we
    // do not need to explicitly check for the extra cases here for use_device
    // addr/ptr
    if (!mapClauseOwner)
      return;

    auto addOperands = [&](aiir::MutableOperandRange &mutableOpRange,
                           aiir::Operation *directiveOp,
                           unsigned blockArgInsertIndex = 0) {
      if (!llvm::is_contained(mutableOpRange.getAsOperandRange(),
                              op.getResult()))
        return;

      // There doesn't appear to be a simple way to convert MutableOperandRange
      // to a vector currently, so we instead use a for_each to populate our
      // vector.
      llvm::SmallVector<aiir::Value> newMapOps;
      newMapOps.reserve(mutableOpRange.size());
      llvm::for_each(
          mutableOpRange.getAsOperandRange(),
          [&newMapOps](aiir::Value oper) { newMapOps.push_back(oper); });

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
        llvm::dyn_cast<aiir::omp::BlockArgOpenMPOpInterface>(target);

    if (auto mapClauseOwner =
            llvm::dyn_cast<aiir::omp::MapClauseOwningOpInterface>(target)) {
      aiir::MutableOperandRange mapMutableOpRange =
          mapClauseOwner.getMapVarsMutable();
      unsigned blockArgInsertIndex =
          argIface
              ? argIface.getMapBlockArgsStart() + argIface.numMapBlockArgs()
              : 0;
      addOperands(mapMutableOpRange,
                  llvm::dyn_cast_if_present<aiir::omp::TargetOp>(
                      argIface.getOperation()),
                  blockArgInsertIndex);
    }

    if (auto targetDataOp = llvm::dyn_cast<aiir::omp::TargetDataOp>(target)) {
      aiir::MutableOperandRange useDevAddrMutableOpRange =
          targetDataOp.getUseDeviceAddrVarsMutable();
      addOperands(useDevAddrMutableOpRange, target,
                  argIface.getUseDeviceAddrBlockArgsStart() +
                      argIface.numUseDeviceAddrBlockArgs());

      aiir::MutableOperandRange useDevPtrMutableOpRange =
          targetDataOp.getUseDevicePtrVarsMutable();
      addOperands(useDevPtrMutableOpRange, target,
                  argIface.getUseDevicePtrBlockArgsStart() +
                      argIface.numUseDevicePtrBlockArgs());
    } else if (auto targetOp = llvm::dyn_cast<aiir::omp::TargetOp>(target)) {
      aiir::MutableOperandRange hasDevAddrMutableOpRange =
          targetOp.getHasDeviceAddrVarsMutable();
      addOperands(hasDevAddrMutableOpRange, target,
                  argIface.getHasDeviceAddrBlockArgsStart() +
                      argIface.numHasDeviceAddrBlockArgs());
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
  aiir::Operation *getFirstTargetUser(aiir::omp::MapInfoOp mapOp) {
    for (auto *user : mapOp->getUsers()) {
      if (llvm::isa<aiir::omp::TargetOp, aiir::omp::TargetDataOp,
                    aiir::omp::TargetUpdateOp, aiir::omp::TargetExitDataOp,
                    aiir::omp::TargetEnterDataOp,
                    aiir::omp::DeclareMapperInfoOp>(user))
        return user;

      if (auto mapUser = llvm::dyn_cast<aiir::omp::MapInfoOp>(user))
        return getFirstTargetUser(mapUser);
    }

    return nullptr;
  }

  void addImplicitDescriptorMapToTargetDataOp(aiir::omp::MapInfoOp op,
                                              fir::FirOpBuilder &builder,
                                              aiir::Operation &target) {
    // Checks if the map is present as an explicit map already on the target
    // data directive, and not just present on a use_device_addr/ptr, as if
    // that's the case, we should not need to add an implicit map for the
    // descriptor.
    auto explicitMappingPresent = [](aiir::omp::MapInfoOp op,
                                     aiir::omp::TargetDataOp tarData) {
      // Verify top-level descriptor mapping is at least equal with same
      // varPtr, the map type should always be To for a descriptor, which is
      // all we really care about for this mapping as we aim to make sure the
      // descriptor is always present on device if we're expecting to access
      // the underlying data.
      if (tarData.getMapVars().empty())
        return false;

      for (aiir::Value mapVar : tarData.getMapVars()) {
        auto mapOp = llvm::cast<aiir::omp::MapInfoOp>(mapVar.getDefiningOp());
        if (mapOp.getVarPtr() == op.getVarPtr() &&
            mapOp.getVarPtrPtr() == op.getVarPtrPtr()) {
          return true;
        }
      }

      return false;
    };

    // if we're not a top level descriptor with members (e.g. member of a
    // derived type), we do not want to perform this step.
    if (!llvm::isa<aiir::omp::TargetDataOp>(target) || op.getMembers().empty())
      return;

    if (!isUseDeviceAddr(op, target) && !isUseDevicePtr(op, target))
      return;

    auto targetDataOp = llvm::cast<aiir::omp::TargetDataOp>(target);
    if (explicitMappingPresent(op, targetDataOp))
      return;

    aiir::omp::MapInfoOp newDescParentMapOp = aiir::omp::MapInfoOp::create(
        builder, op->getLoc(), op.getResult().getType(), op.getVarPtr(),
        op.getVarTypeAttr(),
        builder.getAttr<aiir::omp::ClauseMapFlagsAttr>(
            aiir::omp::ClauseMapFlags::to | aiir::omp::ClauseMapFlags::always),
        op.getMapCaptureTypeAttr(), /*varPtrPtr=*/aiir::Value{},
        aiir::SmallVector<aiir::Value>{}, aiir::ArrayAttr{},
        /*bounds=*/aiir::SmallVector<aiir::Value>{},
        /*mapperId*/ aiir::FlatSymbolRefAttr(), op.getNameAttr(),
        /*partial_map=*/builder.getBoolAttr(false));

    targetDataOp.getMapVarsMutable().append({newDescParentMapOp});
  }

  void removeTopLevelDescriptor(aiir::omp::MapInfoOp op,
                                fir::FirOpBuilder &builder,
                                aiir::Operation *target) {
    if (llvm::isa<aiir::omp::TargetOp, aiir::omp::TargetDataOp,
                  aiir::omp::DeclareMapperInfoOp>(target))
      return;

    // if we're not a top level descriptor with members (e.g. member of a
    // derived type), we do not want to perform this step.
    if (op.getMembers().empty())
      return;

    aiir::SmallVector<aiir::Value> members = op.getMembers();
    aiir::omp::MapInfoOp baseAddr =
        aiir::dyn_cast_or_null<aiir::omp::MapInfoOp>(
            members.front().getDefiningOp());
    assert(baseAddr && "Expected member to be MapInfoOp");
    members.erase(members.begin());

    llvm::SmallVector<llvm::SmallVector<int64_t>> memberIndices;
    getMemberIndicesAsVectors(op, memberIndices);

    // Can skip the extra processing if there's only 1 member as it'd
    // be the base addresses, which we're promoting to the parent.
    aiir::ArrayAttr membersAttr;
    if (memberIndices.size() > 1) {
      memberIndices.erase(memberIndices.begin());
      membersAttr = builder.create2DI64ArrayAttr(memberIndices);
    }

    // VarPtrPtr is tied to detecting if something is a pointer in the later
    // lowering currently, this at the moment comes tied with
    // OMP_MAP_PTR_AND_OBJ being applied which breaks the problem this tries to
    // solve by emitting a 8-byte mapping tied to the descriptor address (even
    // if we only emit a single map). So we circumvent this by removing the
    // varPtrPtr mapping, however, a side affect of this is we lose the
    // additional load from the backend tied to this which is required for
    // correctness and getting the correct address of the data to perform our
    // mapping. So we do our load at this stage.
    // TODO/FIXME: Tidy up the OMP_MAP_PTR_AND_OBJ and varPtrPtr being tied to
    // if something is a pointer to try and tidy up the implementation a bit.
    // This is an unfortunate complexity from push-back from upstream. We
    // could also emit a load at this level for all base addresses as well,
    // which in turn will simplify the later lowering a bit as well. But first
    // need to see how well this alteration works.
    auto loadBaseAddr =
        builder.loadIfRef(op->getLoc(), baseAddr.getVarPtrPtr());
    aiir::omp::MapInfoOp newBaseAddrMapOp = aiir::omp::MapInfoOp::create(
        builder, op->getLoc(), loadBaseAddr.getType(), loadBaseAddr,
        baseAddr.getVarTypeAttr(), baseAddr.getMapTypeAttr(),
        baseAddr.getMapCaptureTypeAttr(), aiir::Value{}, members, membersAttr,
        baseAddr.getBounds(),
        /*mapperId*/ aiir::FlatSymbolRefAttr(), op.getNameAttr(),
        /*partial_map=*/builder.getBoolAttr(false));
    op.replaceAllUsesWith(newBaseAddrMapOp.getResult());
    op->erase();
    baseAddr.erase();
  }

  static bool hasADescriptor(aiir::Operation *varOp, aiir::Type varType) {
    if (fir::isTypeWithDescriptor(varType) ||
        aiir::isa<fir::BoxCharType>(varType) ||
        aiir::isa_and_present<fir::BoxAddrOp>(varOp))
      return true;
    return false;
  }

  // This pass executes on omp::MapInfoOp's containing descriptor based types
  // (allocatables, pointers, assumed shape etc.) and expanding them into
  // multiple omp::MapInfoOp's for each pointer member contained within the
  // descriptor.
  //
  // From the perspective of the AIIR pass manager this runs on the top level
  // operation (usually function) containing the MapInfoOp because this pass
  // will mutate siblings of MapInfoOp.
  void runOnOperation() override {
    aiir::ModuleOp module = getOperation();
    if (!module)
      module = getOperation()->getParentOfType<aiir::ModuleOp>();
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
    getOperation()->walk([&](aiir::Operation *func) {
      if (!aiir::isa<aiir::func::FuncOp, aiir::omp::DeclareMapperOp>(func))
        return;
      // clear all local allocations we made for any boxes in any prior
      // iterations from previous function scopes.
      localBoxAllocas.clear();
      deferrableDesc.clear();

      // Next, walk `omp.map.info` ops to see if any record members should be
      // implicitly mapped.
      func->walk([&](aiir::omp::MapInfoOp op) {
        aiir::Type underlyingType =
            fir::unwrapRefType(op.getVarPtr().getType());

        // TODO Test with and support more complicated cases; like arrays for
        // records, for example.
        if (!fir::isRecordWithAllocatableMember(underlyingType))
          return aiir::WalkResult::advance();

        // TODO For now, only consider `omp.target` ops. Other ops that support
        // `map` clauses will follow later.
        aiir::omp::TargetOp target =
            aiir::dyn_cast_if_present<aiir::omp::TargetOp>(
                getFirstTargetUser(op));

        if (!target)
          return aiir::WalkResult::advance();

        auto mapClauseOwner =
            llvm::dyn_cast<aiir::omp::MapClauseOwningOpInterface>(*target);

        int64_t mapVarIdx = mapClauseOwner.getOperandIndexForMap(op);
        assert(mapVarIdx >= 0 &&
               mapVarIdx <
                   static_cast<int64_t>(mapClauseOwner.getMapVars().size()));

        auto argIface =
            llvm::dyn_cast<aiir::omp::BlockArgOpenMPOpInterface>(*target);
        // TODO How should `map` block argument that correspond to: `private`,
        // `use_device_addr`, `use_device_ptr`, be handled?
        aiir::BlockArgument opBlockArg = argIface.getMapBlockArgs()[mapVarIdx];
        llvm::SetVector<aiir::Operation *> mapVarForwardSlice;
        aiir::getForwardSlice(opBlockArg, &mapVarForwardSlice);

        mapVarForwardSlice.remove_if([&](aiir::Operation *sliceOp) {
          // TODO Support coordinate_of ops.
          //
          // TODO Support call ops by recursively examining the forward slice of
          // the corresponding parameter to the field in the called function.
          return !aiir::isa<hlfir::DesignateOp>(sliceOp);
        });

        auto recordType = aiir::cast<fir::RecordType>(underlyingType);
        llvm::SmallVector<aiir::Value> newMapOpsForFields;
        llvm::SmallVector<llvm::SmallVector<int64_t>> newMemberIndexPaths;

        // 1) Handle direct top-level allocatable fields.
        for (auto fieldMemTyPair : recordType.getTypeList()) {
          auto &field = fieldMemTyPair.first;
          auto memTy = fieldMemTyPair.second;

          if (!fir::isAllocatableType(memTy))
            continue;

          bool referenced = llvm::any_of(mapVarForwardSlice, [&](auto *opv) {
            auto designateOp = aiir::dyn_cast<hlfir::DesignateOp>(opv);
            return designateOp && designateOp.getComponent() &&
                   designateOp.getComponent()->strref() == field;
          });
          if (!referenced)
            continue;

          int32_t fieldIdx = recordType.getFieldIndex(field);
          builder.setInsertionPoint(op);
          fir::IntOrValue idxConst =
              aiir::IntegerAttr::get(builder.getI32Type(), fieldIdx);
          auto fieldCoord = fir::CoordinateOp::create(
              builder, op.getLoc(), builder.getRefType(memTy), op.getVarPtr(),
              llvm::SmallVector<fir::IntOrValue, 1>{idxConst});
          int64_t fieldIdx64 = static_cast<int64_t>(fieldIdx);
          llvm::SmallVector<int64_t, 1> idxPath{fieldIdx64};
          appendMemberMapIfNew(op, builder, op.getLoc(), fieldCoord, idxPath,
                               field, newMapOpsForFields, newMemberIndexPaths);
        }

        // Handle nested allocatable fields along any component chain
        // referenced in the region via HLFIR designates.
        llvm::SmallVector<llvm::SmallVector<int64_t>> seenIndexPaths;
        for (aiir::Operation *sliceOp : mapVarForwardSlice) {
          auto designateOp = aiir::dyn_cast<hlfir::DesignateOp>(sliceOp);
          if (!designateOp || !designateOp.getComponent())
            continue;
          llvm::SmallVector<llvm::StringRef> compPathReversed;
          compPathReversed.push_back(designateOp.getComponent()->strref());
          aiir::Value curBase = designateOp.getMemref();
          bool rootedAtMapArg = false;
          while (true) {
            if (auto parentDes = curBase.getDefiningOp<hlfir::DesignateOp>()) {
              if (!parentDes.getComponent())
                break;
              compPathReversed.push_back(parentDes.getComponent()->strref());
              curBase = parentDes.getMemref();
              continue;
            }
            if (auto decl = curBase.getDefiningOp<hlfir::DeclareOp>()) {
              if (auto barg =
                      aiir::dyn_cast<aiir::BlockArgument>(decl.getMemref()))
                rootedAtMapArg = (barg == opBlockArg);
            } else if (auto blockArg =
                           aiir::dyn_cast_or_null<aiir::BlockArgument>(
                               curBase)) {
              rootedAtMapArg = (blockArg == opBlockArg);
            }
            break;
          }
          // Only process nested paths (2+ components). Single-component paths
          // for direct fields are handled above.
          if (!rootedAtMapArg || compPathReversed.size() < 2)
            continue;
          builder.setInsertionPoint(op);
          llvm::SmallVector<int64_t> indexPath;
          aiir::Type curTy = underlyingType;
          aiir::Value coordRef = op.getVarPtr();
          bool validPath = true;
          for (llvm::StringRef compName : llvm::reverse(compPathReversed)) {
            auto recTy = aiir::dyn_cast<fir::RecordType>(curTy);
            if (!recTy) {
              validPath = false;
              break;
            }
            int32_t idx = recTy.getFieldIndex(compName);
            if (idx < 0) {
              validPath = false;
              break;
            }
            indexPath.push_back(idx);
            aiir::Type memTy = recTy.getType(idx);
            fir::IntOrValue idxConst =
                aiir::IntegerAttr::get(builder.getI32Type(), idx);
            coordRef = fir::CoordinateOp::create(
                builder, op.getLoc(), builder.getRefType(memTy), coordRef,
                llvm::SmallVector<fir::IntOrValue, 1>{idxConst});
            curTy = memTy;
          }
          if (!validPath)
            continue;
          if (auto finalRefTy =
                  aiir::dyn_cast<fir::ReferenceType>(coordRef.getType())) {
            aiir::Type eleTy = finalRefTy.getElementType();
            if (fir::isAllocatableType(eleTy)) {
              if (!containsPath(seenIndexPaths, indexPath)) {
                seenIndexPaths.emplace_back(indexPath.begin(), indexPath.end());
                appendMemberMapIfNew(op, builder, op.getLoc(), coordRef,
                                     indexPath, compPathReversed.front(),
                                     newMapOpsForFields, newMemberIndexPaths);
              }
            }
          }
        }

        if (newMapOpsForFields.empty())
          return aiir::WalkResult::advance();

        // Deduplicate by index path to avoid emitting duplicate members for
        // the same component. Use a set-based key to keep this near O(n).
        llvm::SmallVector<aiir::Value> dedupMapOps;
        llvm::SmallVector<llvm::SmallVector<int64_t>> dedupIndexPaths;
        llvm::StringSet<> seenKeys;
        for (auto [i, mapOp] : llvm::enumerate(newMapOpsForFields)) {
          const auto &path = newMemberIndexPaths[i];
          llvm::SmallString<64> key;
          buildPathKey(path, key);
          if (seenKeys.contains(key))
            continue;
          seenKeys.insert(key);
          dedupMapOps.push_back(mapOp);
          dedupIndexPaths.emplace_back(path.begin(), path.end());
        }
        op.getMembersMutable().append(dedupMapOps);
        llvm::SmallVector<llvm::SmallVector<int64_t>> newMemberIndices;
        if (aiir::ArrayAttr oldAttr = op.getMembersIndexAttr())
          for (aiir::Attribute indexList : oldAttr) {
            llvm::SmallVector<int64_t> listVec;

            for (aiir::Attribute index : aiir::cast<aiir::ArrayAttr>(indexList))
              listVec.push_back(aiir::cast<aiir::IntegerAttr>(index).getInt());

            newMemberIndices.emplace_back(std::move(listVec));
          }
        for (auto &path : dedupIndexPaths)
          newMemberIndices.emplace_back(path);

        op.setMembersIndexAttr(builder.create2DI64ArrayAttr(newMemberIndices));
        // Set to partial map only if there is no user-defined mapper.
        op.setPartialMap(op.getMapperIdAttr() == nullptr);

        return aiir::WalkResult::advance();
      });

      // Expand type(C_PTR) only when unified_shared_memory is required,
      // to ensure device-visible pointer size/behavior in USM scenarios
      // without changing default expectations elsewhere.
      func->walk([&](aiir::omp::MapInfoOp op) {
        // Only expand C_PTR members when unified_shared_memory is required.
        if (!moduleRequiresUSM(func->getParentOfType<aiir::ModuleOp>()))
          return;
        builder.setInsertionPoint(op);
        genCptrMemberMap(op, builder);
      });

      func->walk([&](aiir::omp::MapInfoOp op) {
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

        if (hasADescriptor(op.getVarPtr().getDefiningOp(),
                           fir::unwrapRefType(op.getVarType()))) {
          builder.setInsertionPoint(op);
          aiir::Operation *targetUser = getFirstTargetUser(op);
          assert(targetUser && "expected user of map operation was not found");
          genDescriptorMemberMaps(op, builder, targetUser);
        }
      });

      func->walk([&](aiir::omp::MapInfoOp op) {
        // If a record type is not mapped with the `close` modifier while some
        // of its members are (e.g. descriptor maps), then in USM mode, the
        // memory for the record will be allocated in unified memory while the
        // the members might be allocated in device memory. This creates an
        // inconsistent map for the record type where some of its members are
        // allocated in different address spaces.
        //
        // This fixes this issue by taking a conservative approach and removing
        // the `close` flag from members if it is not used for mapping the
        // parent record.
        if (op.getMembers().empty())
          return;

        aiir::Type varTy = fir::unwrapRefType(op.getVarPtr().getType());
        if (!aiir::isa<fir::RecordType>(varTy))
          return;

        auto mapFlag = op.getMapType();
        bool hasClose = (mapFlag & aiir::omp::ClauseMapFlags::close) ==
                        aiir::omp::ClauseMapFlags::close;

        if (hasClose)
          return;

        for (auto member : op.getMembers()) {
          if (auto memberOp = llvm::dyn_cast_if_present<aiir::omp::MapInfoOp>(
                  member.getDefiningOp())) {
            auto memberMapFlag =
                memberOp.getMapType() & ~aiir::omp::ClauseMapFlags::close;
            memberOp.setMapType(memberMapFlag);
          }
        }
      });

      // Now that we've expanded all of our boxes into a descriptor and base
      // address map where necessary, we check if the map owner is an
      // enter/exit/target data directive, and if they are we drop the initial
      // descriptor (top-level parent) and replace it with the
      // base_address/data.
      //
      // This circumvents issues with stack allocated descriptors bound to
      // device colliding which in Flang is rather trivial for a user to do by
      // accident due to the rather pervasive local intermediate descriptor
      // generation that occurs whenever you pass boxes around different scopes.
      // In OpenMP 6+ mapping these would be a user error as the tools required
      // to circumvent these issues are provided by the spec (ref_ptr/ptee map
      // types), but in prior specifications these tools are not available and
      // it becomes an implementation issue for us to solve.
      //
      // We do this by dropping the top-level descriptor which will be the stack
      // descriptor when we perform enter/exit maps, as we don't want these to
      // be bound until necessary which is when we utilise the descriptor type
      // within a target region. At which point we map the relevant descriptor
      // data and the runtime should correctly associate the data with the
      // descriptor and bind together and allow clean mapping and execution.
      for (auto *op : deferrableDesc) {
        auto mapOp = llvm::dyn_cast<aiir::omp::MapInfoOp>(op);
        aiir::Operation *targetUser = getFirstTargetUser(mapOp);
        assert(targetUser && "expected user of map operation was not found");
        builder.setInsertionPoint(mapOp);
        removeTopLevelDescriptor(mapOp, builder, targetUser);
        addImplicitDescriptorMapToTargetDataOp(mapOp, builder, *targetUser);
      }

      // Wait until after we have generated all of our maps to add them onto
      // the target's block arguments, simplifying the process as there would be
      // no need to avoid accidental duplicate additions.
      func->walk([&](aiir::omp::MapInfoOp op) {
        aiir::Operation *targetUser = getFirstTargetUser(op);
        assert(targetUser && "expected user of map operation was not found");
        addImplicitMembersToTarget(op, builder, targetUser);
      });
    });
  }
};

} // namespace

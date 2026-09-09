//===- ACCMapInfoPrep.cpp - Materialize acc.map_info for FIR --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass replaces the OpenACC data clause operations on FIR-typed operands
// with acc.map_info. A map entry states everything the offload runtime needs
// about one mapped object: the address to transfer, the pointer slot to attach
// it to, the Fortran descriptor that describes it, the element and object
// sizes, the bounds of the section being mapped, and the map-type flags. All
// of that is derived from FIR types and attributes here, so that lowering to
// runtime calls can work from the map entry alone.
//
// A data entry operation and the data exit operations paired with it describe
// the same object, so they collapse into a single map entry whose flags carry
// the effects of both directions. Privatized storage (acc.privatize,
// acc.firstprivate_map) is wrapped the same way, with the parallel levels that
// govern its replication.
//
// Example transformation, for an allocatable scalar in a copy clause:
//
//   Before:
//     %slot = fir.declare %alloca : !fir.ref<!fir.box<!fir.heap<i32>>>
//     %in = acc.copyin varPtr(%slot : !fir.ref<!fir.box<!fir.heap<i32>>>)
//         dataClause(acc_copy) name("n") -> !fir.ref<!fir.box<!fir.heap<i32>>>
//     acc.data dataOperands(%in : !fir.ref<!fir.box<!fir.heap<i32>>>) {
//       ...
//     }
//     acc.copyout accPtr(%in : !fir.ref<!fir.box<!fir.heap<i32>>>)
//         to varPtr(%slot : !fir.ref<!fir.box<!fir.heap<i32>>>)
//         dataClause(acc_copy) name("n")
//
//   After:
//     %slot = fir.declare %alloca : !fir.ref<!fir.box<!fir.heap<i32>>>
//     %c0 = arith.constant 0 : i64
//     // The copyin and the copyout fold into one entry, so the flags name both
//     // directions. The descriptor makes this an attach (ptr_and_obj) of a
//     // CFI-described object, and a size of zero defers the byte count to that
//     // descriptor. exitLoc points at the erased copyout.
//     %map = acc.map_info varPtr(%slot : !fir.ref<!fir.box<!fir.heap<i32>>>)
//         size(%c0 : i64) elementSize(4) name("n") exitLoc(...)
//         descKind(cfi) mapFlags(to,from,ptr_and_obj)
//         -> !fir.ref<!fir.box<!fir.heap<i32>>>
//     acc.data dataOperands(%map : !fir.ref<!fir.box<!fir.heap<i32>>>) {
//       ...
//     }
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Dialect/CUF/Attributes/CUFAttr.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/OpenACC/Analysis/FIROpenACCSupportAnalysis.h"
#include "flang/Optimizer/OpenACC/Passes.h"
#include "flang/Optimizer/OpenACC/Support/FIROpenACCUtils.h"
#include "flang/Optimizer/Support/InternalNames.h"
#include "flang/Semantics/runtime-type-info.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/OpenACC/Analysis/OpenACCSupport.h"
#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "mlir/Dialect/OpenACC/OpenACCUtilsCG.h"
#include "mlir/Dialect/OpenACC/OpenACCUtilsType.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"

namespace fir {
namespace acc {
#define GEN_PASS_DEF_ACCMAPINFOPREP
#include "flang/Optimizer/OpenACC/Passes.h.inc"
} // namespace acc
} // namespace fir

using namespace mlir;

namespace {

/// Returns the pointer slot holding the address of \p mapVar, which the runtime
/// rewrites once the pointee has a device copy. Such a slot exists only when
/// the clause maps a pointee obtained by dereferencing it; mapping the slot
/// itself has no second indirection and therefore no attach point. Only slots
/// reached through a Fortran descriptor are recognized here.
static Value findAttachPoint(Value mapVar) {
  if (auto boxAddr = mapVar.getDefiningOp<fir::BoxAddrOp>()) {
    if (auto load = boxAddr.getVal().getDefiningOp<fir::LoadOp>())
      return load.getMemref();
  }
  if (fir::isa_box_type(fir::unwrapRefType(mapVar.getType()))) {
    if (auto load = mapVar.getDefiningOp<fir::LoadOp>())
      return load.getMemref();
  }
  return {};
}

/// True when the clause maps descriptor storage rather than a plain object. The
/// runtime must then treat the entry as pointer-and-object: besides the
/// descriptor bytes it fixes up the base address recorded inside them.
///
/// The base-address slot is identified downstream by the descriptor address
/// (`desc`) alone, which currently works only because `base_addr` sits at
/// offset 0 of the F18/CFI descriptor, so `&desc == &desc->base_addr`. If that
/// layout changes, the attach point must be formed from the descriptor's actual
/// `base_addr` field rather than the descriptor address.
static bool mapsDescriptorStorage(Value mapVar) {
  auto refTy = dyn_cast<fir::ReferenceType>(mapVar.getType());
  return refTy && fir::isa_box_type(refTy.getEleTy());
}

static bool isManagedData(Value var) {
  auto hasManagedAttr = [](Value v) {
    Operation *op = v.getDefiningOp();
    return op && cuf::hasDataAttr(op, cuf::DataAttribute::Managed);
  };
  if (hasManagedAttr(var))
    return true;
  Value orig = fir::acc::getOriginalDef(var, /*stripDeclare=*/false);
  return orig && orig != var && hasManagedAttr(orig);
}

static std::pair<acc::DataDescKind, Value>
findDescriptorFacts(Value mapVar, Type mappedObjectType, bool isImplicit) {
  Type mapTy = mapVar.getType();
  if (auto refTy = dyn_cast<fir::ReferenceType>(mapTy)) {
    if (fir::isa_box_type(refTy.getEleTy()))
      return {acc::DataDescKind::cfi, mapVar};
  }
  if (fir::isa_box_type(fir::unwrapRefType(mapTy)))
    return {acc::DataDescKind::cfi, mapVar};
  // box_addr of a loaded box can be either the pointee of a nested descriptor
  // map or a host data-base address derived from an already-mapped box. The
  // latter is always an implicit clause; only treat the explicit case as CFI.
  if (!isImplicit) {
    if (auto boxAddr = mapVar.getDefiningOp<fir::BoxAddrOp>()) {
      Value boxVal = boxAddr.getVal();
      if (fir::isa_box_type(boxVal.getType()) &&
          boxVal.getDefiningOp<fir::LoadOp>())
        return {acc::DataDescKind::cfi, boxVal};
    }
  }
  (void)mappedObjectType;
  return {acc::DataDescKind::none, {}};
}

/// Byte size of \p type as storage - what the type alone describes, without a
/// value to interpret it as a mapped object. The types here are mostly FIR
/// ones, which only the OpenACCSupport implementation sizes; it falls back to
/// the dialect-agnostic acc::getTypeSizeAndAlignment for the rest.
static std::optional<int64_t> computeTypeSizeBytes(acc::OpenACCSupport &support,
                                                   ModuleOp module, Type type) {
  std::optional<acc::TypeSizeAndAlignment> sizeAndAlignment =
      support.getTypeSizeAndAlignment(type, module);
  if (!sizeAndAlignment || sizeAndAlignment->first.isScalable())
    return std::nullopt;
  return static_cast<int64_t>(sizeAndAlignment->first.getFixedValue());
}

/// Load the size-in-bytes field from the Fortran type descriptor for
/// \p recordType. Returns null when no matching type-descriptor global is
/// present.
static Value loadRecordTypeSizeFromTypeDesc(
    Location loc, fir::RecordType recordType, Operation *entryOp,
    std::optional<SymbolTable> &symbolTable, OpBuilder &builder) {
  ModuleOp module = entryOp->getParentOfType<ModuleOp>();
  if (!module)
    return {};

  // Keep a TypeDesc use so later passes see the record as referenced.
  (void)fir::TypeDescOp::create(builder, loc, TypeAttr::get(recordType));

  if (!symbolTable)
    symbolTable.emplace(module);
  StringAttr typeDescName = builder.getStringAttr(
      fir::NameUniquer::getTypeDescriptorAssemblyName(recordType.getName()));
  auto global = symbolTable->lookup<fir::GlobalOp>(typeDescName);
  if (!global)
    return {};

  auto typeDescRecTy = dyn_cast<fir::RecordType>(global.getType());
  if (!typeDescRecTy)
    return {};

  Value typeDescAddr = fir::AddrOfOp::create(
      builder, loc, fir::ReferenceType::get(typeDescRecTy), global.getSymbol());
  Type fieldTy = fir::FieldType::get(builder.getContext());
  Value field = fir::FieldIndexOp::create(
      builder, loc, fieldTy, Fortran::semantics::sizeInBytesCompName,
      typeDescRecTy, ValueRange{});
  Type coorTy = fir::ReferenceType::get(
      typeDescRecTy.getType(Fortran::semantics::sizeInBytesCompName));
  Value addr =
      fir::CoordinateOp::create(builder, loc, coorTy, typeDescAddr, field);
  return fir::LoadOp::create(builder, loc, addr);
}

static Value materializeMapSize(acc::OpenACCSupport &support,
                                Operation *entryOp, Value var, Type varType,
                                acc::DataDescKind descKind, ValueRange bounds,
                                acc::MapFlags mapFlags,
                                std::optional<SymbolTable> &symbolTable,
                                OpBuilder &builder) {
  Location loc = entryOp->getLoc();
  Type i64Ty = builder.getI64Type();

  int64_t staticSize = -1;
  if (std::optional<DataLayout> dl = acc::getDataLayout(entryOp)) {
    // Privatized maps keep the full object ArgSize; AccDataDesc carries the
    // section. Device firstprivate copies still index with the parent lower
    // bound, so a compact section size (or ArgSize 0) is incorrect.
    if (bitEnumContainsAny(mapFlags, acc::MapFlags::private_))
      staticSize = acc::computeMapInfoSizeBytes(
          var, varType, acc::DataDescKind::none, /*bounds=*/{}, *dl, &support);
    else
      staticSize = acc::computeMapInfoSizeBytes(var, varType, descKind, bounds,
                                                *dl, &support);
  }

  // Derived types with descriptor fields often have no compile-time layout
  // size; load the type descriptor's size-in-bytes field instead.
  if (staticSize < 0) {
    if (auto recordType =
            dyn_cast<fir::RecordType>(fir::unwrapRefType(varType))) {
      if (Value dynamicSize = loadRecordTypeSizeFromTypeDesc(
              loc, recordType, entryOp, symbolTable, builder))
        return dynamicSize;
    }
  }

  // An implicit present of an object whose size is not recoverable is only an
  // address lookup. Size 0 matches the present-table entry whatever its
  // extents are, including a zero-sized array, whereas an unknown size does
  // not. An explicit clause keeps the unknown size so that the runtime can
  // report the missing data instead.
  if (staticSize < 0 && bounds.empty() &&
      bitEnumContainsAll(mapFlags,
                         acc::MapFlags::present | acc::MapFlags::implicit))
    staticSize = 0;

  return arith::ConstantIntOp::create(builder, loc, i64Ty, staticSize);
}

/// Describes the storage of \p baseTy - the base type of `acc.private_type` -
/// as an element type whose extents are appended to \p extents. Extents that
/// the type does not encode are `ShapedType::kDynamic` and are supplied by the
/// `acc.privatize` dynamic sizes. Returns a null type when the type does not
/// describe the storage, such as a descriptor that carries its own extents.
static Type getPrivateStorageShape(Type baseTy,
                                   SmallVectorImpl<int64_t> &extents) {
  if (auto memrefTy = dyn_cast<MemRefType>(baseTy)) {
    llvm::append_range(extents, memrefTy.getShape());
    return memrefTy.getElementType();
  }

  Type storageTy = baseTy;
  if (Type eleTy = fir::dyn_cast_ptrOrBoxEleTy(baseTy))
    storageTy = eleTy;
  if (fir::isa_box_type(storageTy))
    return {};

  if (auto seqTy = dyn_cast<fir::SequenceType>(storageTy)) {
    if (seqTy.hasUnknownShape())
      return {};
    llvm::append_range(extents, seqTy.getShape());
    return seqTy.getEleTy();
  }
  return storageTy;
}

/// Materializes the byte size of privatized storage. The extents that the type
/// encodes are sized as an array, which is what applies the padded element
/// stride; the dynamic extents are then multiplied in. Returns null when the
/// size is not obtainable.
static Value materializePrivateStorageSize(
    acc::OpenACCSupport &support, ModuleOp module, acc::PrivatizeOp privatizeOp,
    Type elementType, ArrayRef<int64_t> extents,
    std::optional<SymbolTable> &symbolTable, OpBuilder &builder) {
  Location loc = privatizeOp.getLoc();
  ValueRange dynamicSizes = privatizeOp.getDynamicSizes();

  SmallVector<int64_t> staticExtents;
  for (int64_t extent : extents)
    if (!ShapedType::isDynamic(extent))
      staticExtents.push_back(extent);
  if (extents.size() - staticExtents.size() != dynamicSizes.size())
    return {};

  // Size the extents the type encodes as a FIR array, which is what applies
  // the padded element stride - also for a memref base, since the stride rule
  // does not depend on where the element type comes from. A single element
  // stands in when all extents are dynamic: its size is the stride that those
  // extents multiply.
  Type staticTy = elementType;
  if (!extents.empty()) {
    if (staticExtents.empty())
      staticExtents.push_back(1);
    staticTy = fir::SequenceType::get(staticExtents, elementType);
  }

  Value size;
  if (std::optional<int64_t> staticBytes =
          computeTypeSizeBytes(support, module, staticTy)) {
    size = arith::ConstantIntOp::create(builder, loc, builder.getI64Type(),
                                        *staticBytes);
  } else if (auto recordType = dyn_cast<fir::RecordType>(elementType)) {
    // A derived type whose layout is not computable here carries its padded
    // size in the Fortran type descriptor.
    size = loadRecordTypeSizeFromTypeDesc(
        loc, recordType, privatizeOp.getOperation(), symbolTable, builder);
    if (!size)
      return {};
    int64_t staticExtent = 1;
    for (int64_t extent : staticExtents)
      staticExtent *= extent;
    if (staticExtent != 1) {
      Value extentVal = arith::ConstantIntOp::create(
          builder, loc, size.getType(), staticExtent);
      size = arith::MulIOp::create(builder, loc, size, extentVal);
    }
  } else {
    return {};
  }

  for (Value dynamicSize : dynamicSizes) {
    Value extentVal =
        arith::IndexCastOp::create(builder, loc, size.getType(), dynamicSize);
    size = arith::MulIOp::create(builder, loc, size, extentVal);
  }
  return size;
}

/// Wraps \p privatizeOp so that privatized storage carries offload facts in
/// `acc.map_info` like any other mapped variable, including the parallel
/// levels that select gang/worker/vector private replication.
static std::optional<acc::MapInfoOp> buildPrivatizeMapInfo(
    acc::OpenACCSupport &support, ModuleOp module, acc::PrivatizeOp privatizeOp,
    const acc::ACCToGPUMappingPolicy &policy,
    std::optional<SymbolTable> &symbolTable, OpBuilder &builder) {
  auto privateTy =
      dyn_cast<acc::PrivateType>(privatizeOp.getResult().getType());
  if (!privateTy)
    return std::nullopt;
  Type baseTy = privateTy.getBaseTy();

  SmallVector<int64_t> extents;
  Type elementType = getPrivateStorageShape(baseTy, extents);
  if (!elementType)
    return std::nullopt;

  builder.setInsertionPointAfter(privatizeOp);
  Value size = materializePrivateStorageSize(
      support, module, privatizeOp, elementType, extents, symbolTable, builder);
  if (!size)
    return std::nullopt;

  return acc::MapInfoOp::create(
      builder, privatizeOp.getLoc(), privateTy, privatizeOp.getResult(), baseTy,
      acc::computePrivatizeMapFlags(privatizeOp, policy), /*varPtrPtr=*/{},
      /*desc=*/{}, acc::DataDescKind::none, /*bounds=*/{}, /*name=*/{},
      computeTypeSizeBytes(support, module, elementType), size);
}

static std::optional<acc::MapInfoOp>
buildMapInfo(acc::OpenACCSupport &support, ModuleOp module, Operation *entryOp,
             std::optional<SymbolTable> &symbolTable, OpBuilder &builder) {
  if (!entryOp || isa<acc::MapInfoOp>(entryOp))
    return std::nullopt;
  if (!isa<ACC_DATA_ENTRY_OPS>(entryOp))
    return std::nullopt;

  Value var = acc::getVar(entryOp);
  if (!var)
    var = acc::getVarPtr(entryOp);
  // This pass only materializes map_info for FIR-typed operands.
  if (!var || !fir::isa_fir_type(var.getType()))
    return std::nullopt;

  std::optional<acc::DataClause> clause = acc::getDataClause(entryOp);
  if (!clause)
    return std::nullopt;

  Type varType = acc::getVarType(entryOp);
  if (!varType)
    varType = fir::unwrapRefType(var.getType());

  // Implicit clauses that carry a data address derived from a box are not
  // descriptor maps and must not pick up attach / CFI facts from that box.
  // Mapping descriptor storage directly is unaffected: it stays a
  // pointer-and-object map whether the clause is implicit or explicit.
  const bool isImplicit = acc::getImplicitFlag(entryOp);
  // Preserve an attach point already made explicit on the data entry. Otherwise
  // infer one from an explicit FIR descriptor dereference. Implicit present
  // siblings of a descriptor map deliberately do not infer it: the descriptor
  // map already owns the attach semantics.
  Value attachPoint = acc::getVarPtrPtr(entryOp);
  if (!attachPoint && !isImplicit)
    attachPoint = findAttachPoint(var);

  auto [descKind, desc] = findDescriptorFacts(var, varType, isImplicit);
  // When the mapped var *is* the descriptor, leave `desc` unset and rely on
  // `var` whenever descKind is set. Keep `desc` only when it differs (e.g. a
  // pointee map whose CFI metadata lives in a separate box value).
  if (desc && desc == var)
    desc = {};
  acc::MapFlags mapFlags = acc::computeDataClauseMapFlags(
      entryOp, attachPoint || mapsDescriptorStorage(var));
  if (isManagedData(var))
    mapFlags = mapFlags | acc::MapFlags::managed_devptr;

  Type elementType = fir::getFortranElementType(varType);
  std::optional<int64_t> elementSize =
      computeTypeSizeBytes(support, module, elementType);

  SmallVector<Value> bounds = acc::getBounds(entryOp);
  if (auto seqTy =
          dyn_cast_or_null<fir::SequenceType>(fir::unwrapRefType(varType)))
    acc::populateSourceExtents(bounds, seqTy.getShape(), builder);

  Location loc = entryOp->getLoc();
  Value size = materializeMapSize(support, entryOp, var, varType, descKind,
                                  bounds, mapFlags, symbolTable, builder);

  return acc::MapInfoOp::create(builder, loc, entryOp->getResult(0).getType(),
                                var, varType, mapFlags, attachPoint, desc,
                                descKind, bounds, acc::getVarName(entryOp),
                                elementSize, size);
}

static void materializeMapInfoForEntryOp(
    Operation *entryOp,
    llvm::function_ref<std::optional<acc::MapInfoOp>(Operation *, OpBuilder &)>
        buildMapInfo) {
  if (!entryOp || isa<acc::MapInfoOp>(entryOp))
    return;
  OpBuilder builder(entryOp);
  std::optional<acc::MapInfoOp> mapInfo = buildMapInfo(entryOp, builder);
  if (!mapInfo)
    return;

  // declare_enter uses device_resident to establish a persistent allocation.
  // A kernel use must instead find that existing allocation with PRESENT:
  // treating it as device_resident again would perform declaration-time
  // mapping at every launch rather than diagnose a missing declaration map.
  // Represent the two call sites with distinct map_info operations.
  acc::MapFlags flags = mapInfo->getMapFlags();
  if (bitEnumContainsAny(flags, acc::MapFlags::device_resident)) {
    SmallVector<OpOperand *> kernelUses;
    for (OpOperand &use : entryOp->getResult(0).getUses()) {
      Operation *owner = use.getOwner();
      if (isa<acc::KernelEnvironmentOp>(owner) ||
          owner->getParentOfType<acc::KernelEnvironmentOp>())
        kernelUses.push_back(&use);
    }
    if (!kernelUses.empty()) {
      builder.setInsertionPointAfter(*mapInfo);
      acc::MapFlags kernelFlags =
          acc::bitEnumClear(flags, acc::MapFlags::device_resident) |
          acc::MapFlags::present;
      acc::MapInfoOp kernelMap = acc::MapInfoOp::create(
          builder, mapInfo->getLoc(), mapInfo->getAccVar().getType(),
          mapInfo->getVar(), mapInfo->getVarType(), kernelFlags,
          mapInfo->getVarPtrPtr(), mapInfo->getDesc(), mapInfo->getDescKind(),
          mapInfo->getBounds(), acc::getVarName(*mapInfo),
          acc::getMapElementSize(*mapInfo), mapInfo->getSize());
      for (OpOperand *use : kernelUses)
        use->set(kernelMap.getAccVar());
    }
  }

  // The exit clause effects are folded into the map flags, which leaves the
  // paired data exit operations describing nothing that the map entry does not
  // already carry - except where those effects happen, which for a structured
  // construct is its end directive.
  SmallVector<Operation *> exitOps =
      acc::getPairedDataExitOps(entryOp->getResult(0));
  if (!exitOps.empty() && exitOps.front()->getLoc() != mapInfo->getLoc())
    mapInfo->setExitLoc(exitOps.front()->getLoc());
  for (Operation *exitOp : exitOps)
    exitOp->erase();

  entryOp->getResult(0).replaceAllUsesWith(mapInfo->getAccVar());
  entryOp->erase();
}

struct ACCMapInfoPrep
    : public fir::acc::impl::ACCMapInfoPrepBase<ACCMapInfoPrep> {
  void runOnOperation() override {
    FunctionOpInterface func = getOperation();
    ModuleOp module = func->getParentOfType<ModuleOp>();
    if (!module)
      return;

    // FIR type sizes come from the OpenACCSupport implementation registered
    // earlier in the pipeline. Register the FIR one when this pass runs on its
    // own, so that sizing does not silently fall back to the generic handling
    // that knows no FIR type.
    auto cachedAnalysis =
        getCachedParentAnalysis<acc::OpenACCSupport>(func->getParentOp());
    acc::OpenACCSupport *localSupport = nullptr;
    if (!cachedAnalysis) {
      localSupport = &getAnalysis<acc::OpenACCSupport>();
      localSupport->setImplementation(fir::acc::FIROpenACCSupportAnalysis());
    }
    acc::OpenACCSupport &support =
        cachedAnalysis ? cachedAnalysis->get() : *localSupport;
    acc::DefaultACCToGPUMappingPolicy mappingPolicy;

    auto createEntryMapInfo = [&](Operation *entryOp, OpBuilder &builder) {
      return buildMapInfo(support, module, entryOp, symbolTable, builder);
    };

    // Collect before rewriting, which replaces and erases the entry ops.
    SmallVector<Operation *> entryOps;
    SmallVector<acc::PrivatizeOp> privatizeOps;
    func->walk([&](Operation *op) {
      if (auto privatizeOp = dyn_cast<acc::PrivatizeOp>(op)) {
        privatizeOps.push_back(privatizeOp);
        return;
      }
      if (!isa<ACC_DATA_ENTRY_OPS>(op))
        return;
      // acc.use_device stays itself to keep acc.host_data intact, and so does
      // acc.cache for shared memory promotion. Private, firstprivate and
      // reduction storage is created from their recipes; of those clauses only
      // the firstprivate initial value is mapped, as acc.firstprivate_map.
      if (isa<acc::UseDeviceOp, acc::CacheOp, acc::PrivateOp,
              acc::FirstprivateOp, acc::ReductionOp>(op))
        return;
      entryOps.push_back(op);
    });

    for (Operation *entryOp : entryOps)
      materializeMapInfoForEntryOp(entryOp, createEntryMapInfo);

    // Privatized storage is sized on acc.map_info as well. Unlike a data entry
    // op, the privatize op stays: it holds the storage handle and its dynamic
    // sizes.
    for (acc::PrivatizeOp op : privatizeOps) {
      if (llvm::any_of(op.getResult().getUsers(), [](Operation *user) {
            return isa<acc::MapInfoOp>(user);
          }))
        continue;
      OpBuilder builder(op);
      std::optional<acc::MapInfoOp> mapInfo = buildPrivatizeMapInfo(
          support, module, op, mappingPolicy, symbolTable, builder);
      if (!mapInfo)
        continue;
      op.getResult().replaceAllUsesExcept(mapInfo->getAccVar(),
                                          mapInfo->getOperation());
    }
  }

private:
  /// Type-descriptor globals are looked up by name whenever a derived type has
  /// no compile-time layout size. Built on the first such lookup and kept for
  /// later ones, including across the functions this pass instance visits: the
  /// pass creates no module-level symbols, so the table cannot go stale.
  std::optional<SymbolTable> symbolTable;
};

} // namespace

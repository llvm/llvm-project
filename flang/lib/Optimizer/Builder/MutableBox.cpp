//===-- MutableBox.cpp -- MutableBox utilities ----------------------------===//
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

#include "flang/Optimizer/Builder/MutableBox.h"
#include "flang/Optimizer/Builder/Character.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/Runtime/Derived.h"
#include "flang/Optimizer/Builder/Runtime/Stop.h"
#include "flang/Optimizer/Builder/Todo.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIROpsSupport.h"
#include "flang/Optimizer/Support/FatalError.h"

/// Create a fir.box describing the new address, bounds, and length parameters
/// for a MutableBox \p box.
static mlir::Value
createNewFirBox(fir::FirOpBuilder &builder, mlir::Location loc,
                const fir::MutableBoxValue &box, mlir::Value addr,
                mlir::ValueRange lbounds, mlir::ValueRange extents,
                mlir::ValueRange lengths, mlir::Value tdesc = {}) {
  if (addr.getType().isa<fir::BaseBoxType>())
    // The entity is already boxed.
    return builder.createConvert(loc, box.getBoxTy(), addr);

  mlir::Value shape;
  if (!extents.empty()) {
    if (lbounds.empty()) {
      shape = builder.create<fir::ShapeOp>(loc, extents);
    } else {
      llvm::SmallVector<mlir::Value> shapeShiftBounds;
      for (auto [lb, extent] : llvm::zip(lbounds, extents)) {
        shapeShiftBounds.emplace_back(lb);
        shapeShiftBounds.emplace_back(extent);
      }
      auto shapeShiftType =
          fir::ShapeShiftType::get(builder.getContext(), extents.size());
      shape = builder.create<fir::ShapeShiftOp>(loc, shapeShiftType,
                                                shapeShiftBounds);
    }
  } // Otherwise, this a scalar. Leave the shape empty.

  // Ignore lengths if already constant in the box type (this would trigger an
  // error in the embox).
  llvm::SmallVector<mlir::Value> cleanedLengths;
  auto cleanedAddr = addr;
  if (auto charTy = box.getEleTy().dyn_cast<fir::CharacterType>()) {
    // Cast address to box type so that both input and output type have
    // unknown or constant lengths.
    auto bt = box.getBaseTy();
    auto addrTy = addr.getType();
    auto type = addrTy.isa<fir::HeapType>()      ? fir::HeapType::get(bt)
                : addrTy.isa<fir::PointerType>() ? fir::PointerType::get(bt)
                                                 : builder.getRefType(bt);
    cleanedAddr = builder.createConvert(loc, type, addr);
    if (charTy.getLen() == fir::CharacterType::unknownLen())
      cleanedLengths.append(lengths.begin(), lengths.end());
  } else if (box.isDerivedWithLenParameters()) {
    TODO(loc, "updating mutablebox of derived type with length parameters");
    cleanedLengths = lengths;
  }
  mlir::Value emptySlice;
  return builder.create<fir::EmboxOp>(loc, box.getBoxTy(), cleanedAddr, shape,
                                      emptySlice, cleanedLengths, tdesc);
}

//===----------------------------------------------------------------------===//
// MutableBoxValue writer and reader
//===----------------------------------------------------------------------===//

namespace {
/// MutablePropertyWriter and MutablePropertyReader implementations are the only
/// places that depend on how the properties of MutableBoxValue (pointers and
/// allocatables) that can be modified in the lifetime of the entity (address,
/// extents, lower bounds, length parameters) are represented.
/// That is, the properties may be only stored in a fir.box in memory if we
/// need to enforce a single point of truth for the properties across calls.
/// Or, they can be tracked as independent local variables when it is safe to
/// do so. Using bare variables benefits from all optimization passes, even
/// when they are not aware of what a fir.box is and fir.box have not been
/// optimized out yet.

/// MutablePropertyWriter allows reading the properties of a MutableBoxValue.
class MutablePropertyReader {
public:
  MutablePropertyReader(fir::FirOpBuilder &builder, mlir::Location loc,
                        const fir::MutableBoxValue &box,
                        bool forceIRBoxRead = false)
      : builder{builder}, loc{loc}, box{box} {
    if (forceIRBoxRead || !box.isDescribedByVariables())
      irBox = builder.create<fir::LoadOp>(loc, box.getAddr());
  }
  /// Get base address of allocated/associated entity.
  mlir::Value readBaseAddress() {
    if (irBox) {
      auto memrefTy = box.getBoxTy().getEleTy();
      if (!fir::isa_ref_type(memrefTy))
        memrefTy = builder.getRefType(memrefTy);
      return builder.create<fir::BoxAddrOp>(loc, memrefTy, irBox);
    }
    auto addrVar = box.getMutableProperties().addr;
    return builder.create<fir::LoadOp>(loc, addrVar);
  }
  /// Return {lbound, extent} values read from the MutableBoxValue given
  /// the dimension.
  std::pair<mlir::Value, mlir::Value> readShape(unsigned dim) {
    auto idxTy = builder.getIndexType();
    if (irBox) {
      auto dimVal = builder.createIntegerConstant(loc, idxTy, dim);
      auto dimInfo = builder.create<fir::BoxDimsOp>(loc, idxTy, idxTy, idxTy,
                                                    irBox, dimVal);
      return {dimInfo.getResult(0), dimInfo.getResult(1)};
    }
    const auto &mutableProperties = box.getMutableProperties();
    auto lb = builder.create<fir::LoadOp>(loc, mutableProperties.lbounds[dim]);
    auto ext = builder.create<fir::LoadOp>(loc, mutableProperties.extents[dim]);
    return {lb, ext};
  }

  /// Return the character length. If the length was not deferred, the value
  /// that was specified is returned (The mutable fields is not read).
  mlir::Value readCharacterLength() {
    if (box.hasNonDeferredLenParams())
      return box.nonDeferredLenParams()[0];
    if (irBox)
      return fir::factory::CharacterExprHelper{builder, loc}.readLengthFromBox(
          irBox);
    const auto &deferred = box.getMutableProperties().deferredParams;
    if (deferred.empty())
      fir::emitFatalError(loc, "allocatable entity has no length property");
    return builder.create<fir::LoadOp>(loc, deferred[0]);
  }

  /// Read and return all extents. If \p lbounds vector is provided, lbounds are
  /// also read into it.
  llvm::SmallVector<mlir::Value>
  readShape(llvm::SmallVectorImpl<mlir::Value> *lbounds = nullptr) {
    llvm::SmallVector<mlir::Value> extents;
    auto rank = box.rank();
    for (decltype(rank) dim = 0; dim < rank; ++dim) {
      auto [lb, extent] = readShape(dim);
      if (lbounds)
        lbounds->push_back(lb);
      extents.push_back(extent);
    }
    return extents;
  }

  /// Read all mutable properties. Return the base address.
  mlir::Value read(llvm::SmallVectorImpl<mlir::Value> &lbounds,
                   llvm::SmallVectorImpl<mlir::Value> &extents,
                   llvm::SmallVectorImpl<mlir::Value> &lengths) {
    extents = readShape(&lbounds);
    if (box.isCharacter())
      lengths.emplace_back(readCharacterLength());
    else if (box.isDerivedWithLenParameters())
      TODO(loc, "read allocatable or pointer derived type LEN parameters");
    return readBaseAddress();
  }

  /// Return the loaded fir.box.
  mlir::Value getIrBox() const {
    assert(irBox);
    return irBox;
  }

  /// Read the lower bounds
  void getLowerBounds(llvm::SmallVectorImpl<mlir::Value> &lbounds) {
    auto rank = box.rank();
    for (decltype(rank) dim = 0; dim < rank; ++dim)
      lbounds.push_back(std::get<0>(readShape(dim)));
  }

private:
  fir::FirOpBuilder &builder;
  mlir::Location loc;
  fir::MutableBoxValue box;
  mlir::Value irBox;
};

/// MutablePropertyWriter allows modifying the properties of a MutableBoxValue.
class MutablePropertyWriter {
public:
  MutablePropertyWriter(fir::FirOpBuilder &builder, mlir::Location loc,
                        const fir::MutableBoxValue &box,
                        mlir::Value typeSourceBox = {})
      : builder{builder}, loc{loc}, box{box}, typeSourceBox{typeSourceBox} {}
  /// Update MutableBoxValue with new address, shape and length parameters.
  /// Extents and lbounds must all have index type.
  /// lbounds can be empty in which case all ones is assumed.
  /// Length parameters must be provided for the length parameters that are
  /// deferred.
  void updateMutableBox(mlir::Value addr, mlir::ValueRange lbounds,
                        mlir::ValueRange extents, mlir::ValueRange lengths,
                        mlir::Value tdesc = {}) {
    if (box.isDescribedByVariables())
      updateMutableProperties(addr, lbounds, extents, lengths);
    else
      updateIRBox(addr, lbounds, extents, lengths, tdesc);
  }

  /// Update MutableBoxValue with a new fir.box. This requires that the mutable
  /// box is not described by a set of variables, since they could not describe
  /// all that can be described in the new fir.box (e.g. non contiguous entity).
  void updateWithIrBox(mlir::Value newBox) {
    assert(!box.isDescribedByVariables());
    builder.create<fir::StoreOp>(loc, newBox, box.getAddr());
  }
  /// Set unallocated/disassociated status for the entity described by
  /// MutableBoxValue. Deallocation is not performed by this helper.
  void setUnallocatedStatus() {
    if (box.isDescribedByVariables()) {
      auto addrVar = box.getMutableProperties().addr;
      auto nullTy = fir::dyn_cast_ptrEleTy(addrVar.getType());
      builder.create<fir::StoreOp>(loc, builder.createNullConstant(loc, nullTy),
                                   addrVar);
    } else {
      // Note that the dynamic type of polymorphic entities must be reset to the
      // declaration type of the mutable box. See Fortran 2018 7.8.2 NOTE 1.
      // For those, we cannot simply set the address to zero. The way we are
      // currently unallocating fir.box guarantees that we are resetting the
      // type to the declared type. Beware if changing this.
      // Note: the standard is not clear in Deallocate and p => NULL semantics
      // regarding the new dynamic type the entity must have. So far, assume
      // this is just like NULLIFY and the dynamic type must be set to the
      // declared type, not retain the previous dynamic type.
      auto deallocatedBox = fir::factory::createUnallocatedBox(
          builder, loc, box.getBoxTy(), box.nonDeferredLenParams(),
          typeSourceBox);
      builder.create<fir::StoreOp>(loc, deallocatedBox, box.getAddr());
    }
  }

  /// Copy Values from the fir.box into the property variables if any.
  void syncMutablePropertiesFromIRBox() {
    if (!box.isDescribedByVariables())
      return;
    llvm::SmallVector<mlir::Value> lbounds;
    llvm::SmallVector<mlir::Value> extents;
    llvm::SmallVector<mlir::Value> lengths;
    auto addr =
        MutablePropertyReader{builder, loc, box, /*forceIRBoxRead=*/true}.read(
            lbounds, extents, lengths);
    updateMutableProperties(addr, lbounds, extents, lengths);
  }

  /// Copy Values from property variables, if any, into the fir.box.
  void syncIRBoxFromMutableProperties() {
    if (!box.isDescribedByVariables())
      return;
    llvm::SmallVector<mlir::Value> lbounds;
    llvm::SmallVector<mlir::Value> extents;
    llvm::SmallVector<mlir::Value> lengths;
    auto addr = MutablePropertyReader{builder, loc, box}.read(lbounds, extents,
                                                              lengths);
    updateIRBox(addr, lbounds, extents, lengths);
  }

private:
  /// Update the IR box (fir.ref<fir.box<T>>) of the MutableBoxValue.
  void updateIRBox(mlir::Value addr, mlir::ValueRange lbounds,
                   mlir::ValueRange extents, mlir::ValueRange lengths,
                   mlir::Value tdesc = {}) {
    mlir::Value irBox = createNewFirBox(builder, loc, box, addr, lbounds,
                                        extents, lengths, tdesc);
    builder.create<fir::StoreOp>(loc, irBox, box.getAddr());
  }

  /// Update the set of property variables of the MutableBoxValue.
  void updateMutableProperties(mlir::Value addr, mlir::ValueRange lbounds,
                               mlir::ValueRange extents,
                               mlir::ValueRange lengths) {
    auto castAndStore = [&](mlir::Value val, mlir::Value addr) {
      auto type = fir::dyn_cast_ptrEleTy(addr.getType());
      builder.create<fir::StoreOp>(loc, builder.createConvert(loc, type, val),
                                   addr);
    };
    const auto &mutableProperties = box.getMutableProperties();
    castAndStore(addr, mutableProperties.addr);
    for (auto [extent, extentVar] :
         llvm::zip(extents, mutableProperties.extents))
      castAndStore(extent, extentVar);
    if (!mutableProperties.lbounds.empty()) {
      if (lbounds.empty()) {
        auto one =
            builder.createIntegerConstant(loc, builder.getIndexType(), 1);
        for (auto lboundVar : mutableProperties.lbounds)
          castAndStore(one, lboundVar);
      } else {
        for (auto [lbound, lboundVar] :
             llvm::zip(lbounds, mutableProperties.lbounds))
          castAndStore(lbound, lboundVar);
      }
    }
    if (box.isCharacter())
      // llvm::zip account for the fact that the length only needs to be stored
      // when it is specified in the allocation and deferred in the
      // MutableBoxValue.
      for (auto [len, lenVar] :
           llvm::zip(lengths, mutableProperties.deferredParams))
        castAndStore(len, lenVar);
    else if (box.isDerivedWithLenParameters())
      TODO(loc, "update allocatable derived type length parameters");
  }
  fir::FirOpBuilder &builder;
  mlir::Location loc;
  fir::MutableBoxValue box;
  mlir::Value typeSourceBox;
};

} // namespace

mlir::Value fir::factory::createUnallocatedBox(
    fir::FirOpBuilder &builder, mlir::Location loc, mlir::Type boxType,
    mlir::ValueRange nonDeferredParams, mlir::Value typeSourceBox) {
  auto baseAddrType = boxType.dyn_cast<fir::BaseBoxType>().getEleTy();
  if (!fir::isa_ref_type(baseAddrType))
    baseAddrType = builder.getRefType(baseAddrType);
  auto type = fir::unwrapRefType(baseAddrType);
  auto eleTy = fir::unwrapSequenceType(type);
  if (auto recTy = eleTy.dyn_cast<fir::RecordType>())
    if (recTy.getNumLenParams() > 0)
      TODO(loc, "creating unallocated fir.box of derived type with length "
                "parameters");
  auto nullAddr = builder.createNullConstant(loc, baseAddrType);
  mlir::Value shape;
  if (auto seqTy = type.dyn_cast<fir::SequenceType>()) {
    auto zero = builder.createIntegerConstant(loc, builder.getIndexType(), 0);
    llvm::SmallVector<mlir::Value> extents(seqTy.getDimension(), zero);
    shape = builder.createShape(
        loc, fir::ArrayBoxValue{nullAddr, extents, /*lbounds=*/std::nullopt});
  }
  // Provide dummy length parameters if they are dynamic. If a length parameter
  // is deferred. It is set to zero here and will be set on allocation.
  llvm::SmallVector<mlir::Value> lenParams;
  if (auto charTy = eleTy.dyn_cast<fir::CharacterType>()) {
    if (charTy.getLen() == fir::CharacterType::unknownLen()) {
      if (!nonDeferredParams.empty()) {
        lenParams.push_back(nonDeferredParams[0]);
      } else {
        auto zero = builder.createIntegerConstant(
            loc, builder.getCharacterLengthType(), 0);
        lenParams.push_back(zero);
      }
    }
  }
  mlir::Value emptySlice;
  return builder.create<fir::EmboxOp>(loc, boxType, nullAddr, shape, emptySlice,
                                      lenParams, typeSourceBox);
}

fir::MutableBoxValue fir::factory::createTempMutableBox(
    fir::FirOpBuilder &builder, mlir::Location loc, mlir::Type type,
    llvm::StringRef name, mlir::Value typeSourceBox) {
  mlir::Type boxType;
  if (typeSourceBox)
    boxType = fir::ClassType::get(fir::HeapType::get(type));
  else
    boxType = fir::BoxType::get(fir::HeapType::get(type));
  auto boxAddr = builder.createTemporary(loc, boxType, name);
  auto box =
      fir::MutableBoxValue(boxAddr, /*nonDeferredParams=*/mlir::ValueRange(),
                           /*mutableProperties=*/{});
  MutablePropertyWriter{builder, loc, box, typeSourceBox}
      .setUnallocatedStatus();
  return box;
}

/// Helper to decide if a MutableBoxValue must be read to a BoxValue or
/// can be read to a reified box value.
static bool readToBoxValue(const fir::MutableBoxValue &box,
                           bool mayBePolymorphic) {
  // If this is described by a set of local variables, the value
  // should not be tracked as a fir.box.
  if (box.isDescribedByVariables())
    return false;
  // Polymorphism might be a source of discontiguity, even on allocatables.
  // Track value as fir.box
  if ((box.isDerived() && mayBePolymorphic) || box.isUnlimitedPolymorphic())
    return true;
  // Intrinsic allocatables are contiguous, no need to track the value by
  // fir.box.
  if (box.isAllocatable() || box.rank() == 0)
    return false;
  // Pointers are known to be contiguous at compile time iff they have the
  // CONTIGUOUS attribute.
  return !fir::valueHasFirAttribute(box.getAddr(),
                                    fir::getContiguousAttrName());
}

fir::ExtendedValue
fir::factory::genMutableBoxRead(fir::FirOpBuilder &builder, mlir::Location loc,
                                const fir::MutableBoxValue &box,
                                bool mayBePolymorphic) {
  if (box.hasAssumedRank())
    TODO(loc, "assumed rank allocatables or pointers");
  llvm::SmallVector<mlir::Value> lbounds;
  llvm::SmallVector<mlir::Value> extents;
  llvm::SmallVector<mlir::Value> lengths;
  if (readToBoxValue(box, mayBePolymorphic)) {
    auto reader = MutablePropertyReader(builder, loc, box);
    reader.getLowerBounds(lbounds);
    return fir::BoxValue{reader.getIrBox(), lbounds,
                         box.nonDeferredLenParams()};
  }
  // Contiguous intrinsic type entity: all the data can be extracted from the
  // fir.box.
  auto addr =
      MutablePropertyReader(builder, loc, box).read(lbounds, extents, lengths);
  auto rank = box.rank();
  if (box.isCharacter()) {
    auto len = lengths.empty() ? mlir::Value{} : lengths[0];
    if (rank)
      return fir::CharArrayBoxValue{addr, len, extents, lbounds};
    return fir::CharBoxValue{addr, len};
  }
  mlir::Value sourceBox;
  if (box.isPolymorphic())
    sourceBox = builder.create<fir::LoadOp>(loc, box.getAddr());
  if (rank)
    return fir::ArrayBoxValue{addr, extents, lbounds, sourceBox};
  if (box.isPolymorphic())
    return fir::PolymorphicValue(addr, sourceBox);
  return addr;
}

mlir::Value
fir::factory::genIsAllocatedOrAssociatedTest(fir::FirOpBuilder &builder,
                                             mlir::Location loc,
                                             const fir::MutableBoxValue &box) {
  auto addr = MutablePropertyReader(builder, loc, box).readBaseAddress();
  return builder.genIsNotNullAddr(loc, addr);
}

/// Generate finalizer call and inlined free. This does not check that the
/// address was allocated.
static void genFinalizeAndFree(fir::FirOpBuilder &builder, mlir::Location loc,
                               mlir::Value addr) {
  // TODO: call finalizer if any.

  // A heap (ALLOCATABLE) object may have been converted to a ptr (POINTER),
  // so make sure the heap type is restored before deallocation.
  auto cast = builder.createConvert(
      loc, fir::HeapType::get(fir::dyn_cast_ptrEleTy(addr.getType())), addr);
  builder.create<fir::FreeMemOp>(loc, cast);
}

void fir::factory::genFinalization(fir::FirOpBuilder &builder,
                                   mlir::Location loc,
                                   const fir::MutableBoxValue &box) {
  auto addr = MutablePropertyReader(builder, loc, box).readBaseAddress();
  auto isAllocated = builder.genIsNotNullAddr(loc, addr);
  auto ifOp = builder.create<fir::IfOp>(loc, isAllocated,
                                        /*withElseRegion=*/false);
  auto insPt = builder.saveInsertionPoint();
  builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
  genFinalizeAndFree(builder, loc, addr);
  builder.restoreInsertionPoint(insPt);
}

//===----------------------------------------------------------------------===//
// MutableBoxValue writing interface implementation
//===----------------------------------------------------------------------===//

void fir::factory::associateMutableBox(fir::FirOpBuilder &builder,
                                       mlir::Location loc,
                                       const fir::MutableBoxValue &box,
                                       const fir::ExtendedValue &source,
                                       mlir::ValueRange lbounds) {
  MutablePropertyWriter writer(builder, loc, box);
  source.match(
      [&](const fir::PolymorphicValue &p) {
        mlir::Value sourceBox;
        if (auto polyBox = source.getBoxOf<fir::PolymorphicValue>())
          sourceBox = polyBox->getSourceBox();
        writer.updateMutableBox(p.getAddr(), /*lbounds=*/std::nullopt,
                                /*extents=*/std::nullopt,
                                /*lengths=*/std::nullopt, sourceBox);
      },
      [&](const fir::UnboxedValue &addr) {
        writer.updateMutableBox(addr, /*lbounds=*/std::nullopt,
                                /*extents=*/std::nullopt,
                                /*lengths=*/std::nullopt);
      },
      [&](const fir::CharBoxValue &ch) {
        writer.updateMutableBox(ch.getAddr(), /*lbounds=*/std::nullopt,
                                /*extents=*/std::nullopt, {ch.getLen()});
      },
      [&](const fir::ArrayBoxValue &arr) {
        writer.updateMutableBox(arr.getAddr(),
                                lbounds.empty() ? arr.getLBounds() : lbounds,
                                arr.getExtents(), /*lengths=*/std::nullopt);
      },
      [&](const fir::CharArrayBoxValue &arr) {
        writer.updateMutableBox(arr.getAddr(),
                                lbounds.empty() ? arr.getLBounds() : lbounds,
                                arr.getExtents(), {arr.getLen()});
      },
      [&](const fir::BoxValue &arr) {
        // Rebox array fir.box to the pointer type and apply potential new lower
        // bounds.
        mlir::ValueRange newLbounds = lbounds.empty()
                                          ? mlir::ValueRange{arr.getLBounds()}
                                          : mlir::ValueRange{lbounds};
        if (box.isDescribedByVariables()) {
          // LHS is a contiguous pointer described by local variables. Open RHS
          // fir.box to update the LHS.
          auto rawAddr = builder.create<fir::BoxAddrOp>(loc, arr.getMemTy(),
                                                        arr.getAddr());
          auto extents = fir::factory::getExtents(loc, builder, source);
          llvm::SmallVector<mlir::Value> lenParams;
          if (arr.isCharacter()) {
            lenParams.emplace_back(
                fir::factory::readCharLen(builder, loc, source));
          } else if (arr.isDerivedWithLenParameters()) {
            TODO(loc, "pointer assignment to derived with length parameters");
          }
          writer.updateMutableBox(rawAddr, newLbounds, extents, lenParams);
        } else {
          mlir::Value shift;
          if (!newLbounds.empty()) {
            auto shiftType =
                fir::ShiftType::get(builder.getContext(), newLbounds.size());
            shift = builder.create<fir::ShiftOp>(loc, shiftType, newLbounds);
          }
          auto reboxed =
              builder.create<fir::ReboxOp>(loc, box.getBoxTy(), arr.getAddr(),
                                           shift, /*slice=*/mlir::Value());
          writer.updateWithIrBox(reboxed);
        }
      },
      [&](const fir::MutableBoxValue &) {
        // No point implementing this, if right-hand side is a
        // pointer/allocatable, the related MutableBoxValue has been read into
        // another ExtendedValue category.
        fir::emitFatalError(loc,
                            "Cannot write MutableBox to another MutableBox");
      },
      [&](const fir::ProcBoxValue &) {
        TODO(loc, "procedure pointer assignment");
      });
}

void fir::factory::associateMutableBoxWithRemap(
    fir::FirOpBuilder &builder, mlir::Location loc,
    const fir::MutableBoxValue &box, const fir::ExtendedValue &source,
    mlir::ValueRange lbounds, mlir::ValueRange ubounds) {
  // Compute new extents
  llvm::SmallVector<mlir::Value> extents;
  auto idxTy = builder.getIndexType();
  if (!lbounds.empty()) {
    auto one = builder.createIntegerConstant(loc, idxTy, 1);
    for (auto [lb, ub] : llvm::zip(lbounds, ubounds)) {
      auto lbi = builder.createConvert(loc, idxTy, lb);
      auto ubi = builder.createConvert(loc, idxTy, ub);
      auto diff = builder.create<mlir::arith::SubIOp>(loc, idxTy, ubi, lbi);
      extents.emplace_back(
          builder.create<mlir::arith::AddIOp>(loc, idxTy, diff, one));
    }
  } else {
    // lbounds are default. Upper bounds and extents are the same.
    for (auto ub : ubounds) {
      auto cast = builder.createConvert(loc, idxTy, ub);
      extents.emplace_back(cast);
    }
  }
  const auto newRank = extents.size();
  auto cast = [&](mlir::Value addr) -> mlir::Value {
    // Cast base addr to new sequence type.
    auto ty = fir::dyn_cast_ptrEleTy(addr.getType());
    if (auto seqTy = ty.dyn_cast<fir::SequenceType>()) {
      fir::SequenceType::Shape shape(newRank,
                                     fir::SequenceType::getUnknownExtent());
      ty = fir::SequenceType::get(shape, seqTy.getEleTy());
    }
    return builder.createConvert(loc, builder.getRefType(ty), addr);
  };
  MutablePropertyWriter writer(builder, loc, box);
  source.match(
      [&](const fir::PolymorphicValue &p) {
        writer.updateMutableBox(cast(p.getAddr()), lbounds, extents,
                                /*lengths=*/std::nullopt);
      },
      [&](const fir::UnboxedValue &addr) {
        writer.updateMutableBox(cast(addr), lbounds, extents,
                                /*lengths=*/std::nullopt);
      },
      [&](const fir::CharBoxValue &ch) {
        writer.updateMutableBox(cast(ch.getAddr()), lbounds, extents,
                                {ch.getLen()});
      },
      [&](const fir::ArrayBoxValue &arr) {
        writer.updateMutableBox(cast(arr.getAddr()), lbounds, extents,
                                /*lengths=*/std::nullopt);
      },
      [&](const fir::CharArrayBoxValue &arr) {
        writer.updateMutableBox(cast(arr.getAddr()), lbounds, extents,
                                {arr.getLen()});
      },
      [&](const fir::BoxValue &arr) {
        // Rebox right-hand side fir.box with a new shape and type.
        if (box.isDescribedByVariables()) {
          // LHS is a contiguous pointer described by local variables. Open RHS
          // fir.box to update the LHS.
          auto rawAddr = builder.create<fir::BoxAddrOp>(loc, arr.getMemTy(),
                                                        arr.getAddr());
          llvm::SmallVector<mlir::Value> lenParams;
          if (arr.isCharacter()) {
            lenParams.emplace_back(
                fir::factory::readCharLen(builder, loc, source));
          } else if (arr.isDerivedWithLenParameters()) {
            TODO(loc, "pointer assignment to derived with length parameters");
          }
          writer.updateMutableBox(rawAddr, lbounds, extents, lenParams);
        } else {
          auto shapeType =
              fir::ShapeShiftType::get(builder.getContext(), extents.size());
          llvm::SmallVector<mlir::Value> shapeArgs;
          auto idxTy = builder.getIndexType();
          for (auto [lbnd, ext] : llvm::zip(lbounds, extents)) {
            auto lb = builder.createConvert(loc, idxTy, lbnd);
            shapeArgs.push_back(lb);
            shapeArgs.push_back(ext);
          }
          auto shape =
              builder.create<fir::ShapeShiftOp>(loc, shapeType, shapeArgs);
          auto reboxed =
              builder.create<fir::ReboxOp>(loc, box.getBoxTy(), arr.getAddr(),
                                           shape, /*slice=*/mlir::Value());
          writer.updateWithIrBox(reboxed);
        }
      },
      [&](const fir::MutableBoxValue &) {
        // No point implementing this, if right-hand side is a pointer or
        // allocatable, the related MutableBoxValue has already been read into
        // another ExtendedValue category.
        fir::emitFatalError(loc,
                            "Cannot write MutableBox to another MutableBox");
      },
      [&](const fir::ProcBoxValue &) {
        TODO(loc, "procedure pointer assignment");
      });
}

void fir::factory::disassociateMutableBox(fir::FirOpBuilder &builder,
                                          mlir::Location loc,
                                          const fir::MutableBoxValue &box,
                                          bool polymorphicSetType) {
  if (box.isPolymorphic() && polymorphicSetType) {
    // 7.3.2.3 point 7. The dynamic type of a disassociated pointer is the
    // same as its declared type.
    auto boxTy = box.getBoxTy().dyn_cast<fir::BaseBoxType>();
    auto eleTy = fir::dyn_cast_ptrOrBoxEleTy(boxTy.getEleTy());
    mlir::Type derivedType = fir::getDerivedType(eleTy);
    if (auto recTy = derivedType.dyn_cast<fir::RecordType>())
      fir::runtime::genNullifyDerivedType(builder, loc, box.getAddr(), recTy,
                                          box.rank());
    return;
  }
  MutablePropertyWriter{builder, loc, box}.setUnallocatedStatus();
}

static llvm::SmallVector<mlir::Value>
getNewLengths(fir::FirOpBuilder &builder, mlir::Location loc,
              const fir::MutableBoxValue &box, mlir::ValueRange lenParams) {
  llvm::SmallVector<mlir::Value> lengths;
  auto idxTy = builder.getIndexType();
  if (auto charTy = box.getEleTy().dyn_cast<fir::CharacterType>()) {
    if (charTy.getLen() == fir::CharacterType::unknownLen()) {
      if (box.hasNonDeferredLenParams()) {
        lengths.emplace_back(
            builder.createConvert(loc, idxTy, box.nonDeferredLenParams()[0]));
      } else if (!lenParams.empty()) {
        mlir::Value len =
            fir::factory::genMaxWithZero(builder, loc, lenParams[0]);
        lengths.emplace_back(builder.createConvert(loc, idxTy, len));
      } else {
        fir::emitFatalError(
            loc, "could not deduce character lengths in character allocation");
      }
    }
  }
  return lengths;
}

static mlir::Value allocateAndInitNewStorage(fir::FirOpBuilder &builder,
                                             mlir::Location loc,
                                             const fir::MutableBoxValue &box,
                                             mlir::ValueRange extents,
                                             mlir::ValueRange lenParams,
                                             llvm::StringRef allocName) {
  auto lengths = getNewLengths(builder, loc, box, lenParams);
  auto newStorage = builder.create<fir::AllocMemOp>(
      loc, box.getBaseTy(), allocName, lengths, extents);
  if (box.getEleTy().isa<fir::RecordType>()) {
    // TODO: skip runtime initialization if this is not required. Currently,
    // there is no way to know here if a derived type needs it or not. But the
    // information is available at compile time and could be reflected here
    // somehow.
    mlir::Value irBox = createNewFirBox(builder, loc, box, newStorage,
                                        std::nullopt, extents, lengths);
    fir::runtime::genDerivedTypeInitialize(builder, loc, irBox);
  }
  return newStorage;
}

void fir::factory::genInlinedAllocation(fir::FirOpBuilder &builder,
                                        mlir::Location loc,
                                        const fir::MutableBoxValue &box,
                                        mlir::ValueRange lbounds,
                                        mlir::ValueRange extents,
                                        mlir::ValueRange lenParams,
                                        llvm::StringRef allocName) {
  auto lengths = getNewLengths(builder, loc, box, lenParams);
  llvm::SmallVector<mlir::Value> safeExtents;
  for (mlir::Value extent : extents)
    safeExtents.push_back(fir::factory::genMaxWithZero(builder, loc, extent));
  auto heap = builder.create<fir::AllocMemOp>(loc, box.getBaseTy(), allocName,
                                              lengths, safeExtents);
  MutablePropertyWriter{builder, loc, box}.updateMutableBox(
      heap, lbounds, safeExtents, lengths);
  if (box.getEleTy().isa<fir::RecordType>()) {
    // TODO: skip runtime initialization if this is not required. Currently,
    // there is no way to know here if a derived type needs it or not. But the
    // information is available at compile time and could be reflected here
    // somehow.
    mlir::Value irBox = fir::factory::getMutableIRBox(builder, loc, box);
    fir::runtime::genDerivedTypeInitialize(builder, loc, irBox);
  }
}

void fir::factory::genInlinedDeallocate(fir::FirOpBuilder &builder,
                                        mlir::Location loc,
                                        const fir::MutableBoxValue &box) {
  auto addr = MutablePropertyReader(builder, loc, box).readBaseAddress();
  genFinalizeAndFree(builder, loc, addr);
  MutablePropertyWriter{builder, loc, box}.setUnallocatedStatus();
}

fir::factory::MutableBoxReallocation fir::factory::genReallocIfNeeded(
    fir::FirOpBuilder &builder, mlir::Location loc,
    const fir::MutableBoxValue &box, mlir::ValueRange shape,
    mlir::ValueRange lengthParams,
    fir::factory::ReallocStorageHandlerFunc storageHandler) {
  // Implement 10.2.1.3 point 3 logic when lhs is an array.
  auto reader = MutablePropertyReader(builder, loc, box);
  auto addr = reader.readBaseAddress();
  auto i1Type = builder.getI1Type();
  auto addrType = addr.getType();
  auto isAllocated = builder.genIsNotNullAddr(loc, addr);
  auto getExtValForStorage = [&](mlir::Value newAddr) -> fir::ExtendedValue {
    mlir::SmallVector<mlir::Value> extents;
    if (box.hasRank()) {
      if (shape.empty())
        extents = reader.readShape();
      else
        extents.append(shape.begin(), shape.end());
    }
    if (box.isCharacter()) {
      auto len = box.hasNonDeferredLenParams() ? reader.readCharacterLength()
                                               : lengthParams[0];
      if (box.hasRank())
        return fir::CharArrayBoxValue{newAddr, len, extents};
      return fir::CharBoxValue{newAddr, len};
    }
    if (box.isDerivedWithLenParameters())
      TODO(loc, "reallocation of derived type entities with length parameters");
    if (box.hasRank())
      return fir::ArrayBoxValue{newAddr, extents};
    return newAddr;
  };
  auto ifOp =
      builder
          .genIfOp(loc, {i1Type, addrType}, isAllocated,
                   /*withElseRegion=*/true)
          .genThen([&]() {
            // The box is allocated. Check if it must be reallocated and
            // reallocate.
            auto mustReallocate = builder.createBool(loc, false);
            auto compareProperty = [&](mlir::Value previous,
                                       mlir::Value required) {
              auto castPrevious =
                  builder.createConvert(loc, required.getType(), previous);
              auto cmp = builder.create<mlir::arith::CmpIOp>(
                  loc, mlir::arith::CmpIPredicate::ne, castPrevious, required);
              mustReallocate = builder.create<mlir::arith::SelectOp>(
                  loc, cmp, cmp, mustReallocate);
            };
            llvm::SmallVector<mlir::Value> previousExtents = reader.readShape();
            if (!shape.empty())
              for (auto [previousExtent, requested] :
                   llvm::zip(previousExtents, shape))
                compareProperty(previousExtent, requested);

            if (box.isCharacter() && !box.hasNonDeferredLenParams()) {
              // When the allocatable length is not deferred, it must not be
              // reallocated in case of length mismatch, instead,
              // padding/trimming will occur in later assignment to it.
              assert(!lengthParams.empty() &&
                     "must provide length parameters for character");
              compareProperty(reader.readCharacterLength(), lengthParams[0]);
            } else if (box.isDerivedWithLenParameters()) {
              TODO(loc, "automatic allocation of derived type allocatable with "
                        "length parameters");
            }
            auto ifOp = builder
                            .genIfOp(loc, {addrType}, mustReallocate,
                                     /*withElseRegion=*/true)
                            .genThen([&]() {
                              // If shape or length mismatch, allocate new
                              // storage. When rhs is a scalar, keep the
                              // previous shape
                              auto extents =
                                  shape.empty()
                                      ? mlir::ValueRange(previousExtents)
                                      : shape;
                              auto heap = allocateAndInitNewStorage(
                                  builder, loc, box, extents, lengthParams,
                                  ".auto.alloc");
                              if (storageHandler)
                                storageHandler(getExtValForStorage(heap));
                              builder.create<fir::ResultOp>(loc, heap);
                            })
                            .genElse([&]() {
                              if (storageHandler)
                                storageHandler(getExtValForStorage(addr));
                              builder.create<fir::ResultOp>(loc, addr);
                            });
            ifOp.end();
            auto newAddr = ifOp.getResults()[0];
            builder.create<fir::ResultOp>(
                loc, mlir::ValueRange{mustReallocate, newAddr});
          })
          .genElse([&]() {
            auto trueValue = builder.createBool(loc, true);
            // The box is not yet allocated, simply allocate it.
            if (shape.empty() && box.rank() != 0) {
              // See 10.2.1.3 p3.
              fir::runtime::genReportFatalUserError(
                  builder, loc,
                  "array left hand side must be allocated when the right hand "
                  "side is a scalar");
              builder.create<fir::ResultOp>(loc,
                                            mlir::ValueRange{trueValue, addr});
            } else {
              auto heap = allocateAndInitNewStorage(
                  builder, loc, box, shape, lengthParams, ".auto.alloc");
              if (storageHandler)
                storageHandler(getExtValForStorage(heap));
              builder.create<fir::ResultOp>(loc,
                                            mlir::ValueRange{trueValue, heap});
            }
          });
  ifOp.end();
  auto wasReallocated = ifOp.getResults()[0];
  auto newAddr = ifOp.getResults()[1];
  // Create an ExtentedValue for the new storage.
  auto newValue = getExtValForStorage(newAddr);
  return {newValue, addr, wasReallocated, isAllocated};
}

void fir::factory::finalizeRealloc(fir::FirOpBuilder &builder,
                                   mlir::Location loc,
                                   const fir::MutableBoxValue &box,
                                   mlir::ValueRange lbounds,
                                   bool takeLboundsIfRealloc,
                                   const MutableBoxReallocation &realloc) {
  builder.genIfThen(loc, realloc.wasReallocated)
      .genThen([&]() {
        auto reader = MutablePropertyReader(builder, loc, box);
        llvm::SmallVector<mlir::Value> previousLbounds;
        if (!takeLboundsIfRealloc && box.hasRank())
          reader.readShape(&previousLbounds);
        auto lbs =
            takeLboundsIfRealloc ? lbounds : mlir::ValueRange{previousLbounds};
        llvm::SmallVector<mlir::Value> lenParams;
        if (box.isCharacter())
          lenParams.push_back(fir::getLen(realloc.newValue));
        if (box.isDerivedWithLenParameters())
          TODO(loc,
               "reallocation of derived type entities with length parameters");
        auto lengths = getNewLengths(builder, loc, box, lenParams);
        auto heap = fir::getBase(realloc.newValue);
        auto extents = fir::factory::getExtents(loc, builder, realloc.newValue);
        builder.genIfThen(loc, realloc.oldAddressWasAllocated)
            .genThen(
                [&]() { genFinalizeAndFree(builder, loc, realloc.oldAddress); })
            .end();
        MutablePropertyWriter{builder, loc, box}.updateMutableBox(
            heap, lbs, extents, lengths);
      })
      .end();
}

//===----------------------------------------------------------------------===//
// MutableBoxValue syncing implementation
//===----------------------------------------------------------------------===//

/// Depending on the implementation, allocatable/pointer descriptor and the
/// MutableBoxValue need to be synced before and after calls passing the
/// descriptor. These calls will generate the syncing if needed or be no-op.
mlir::Value fir::factory::getMutableIRBox(fir::FirOpBuilder &builder,
                                          mlir::Location loc,
                                          const fir::MutableBoxValue &box) {
  MutablePropertyWriter{builder, loc, box}.syncIRBoxFromMutableProperties();
  return box.getAddr();
}
void fir::factory::syncMutableBoxFromIRBox(fir::FirOpBuilder &builder,
                                           mlir::Location loc,
                                           const fir::MutableBoxValue &box) {
  MutablePropertyWriter{builder, loc, box}.syncMutablePropertiesFromIRBox();
}

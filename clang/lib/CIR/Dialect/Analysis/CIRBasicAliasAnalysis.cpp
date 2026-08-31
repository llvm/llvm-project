//===- CIRBasicAliasAnalysis.cpp - Basic CIR Alias Analysis ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/CIR/Dialect/Analysis/CIRBasicAliasAnalysis.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "clang/CIR/Dialect/IR/CIRAttrs.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "llvm/Support/DebugLog.h"
#include "llvm/Support/MathExtras.h"

#include <limits>

#define DEBUG_TYPE "cir-basic-alias-analysis"

using namespace llvm;
using namespace cir;

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

static constexpr unsigned MaxLookupDepth = 6;

/// Return the size in bytes of \p type, or std::nullopt when that size isn't
/// statically known (void, function types, incomplete records, ...).
static std::optional<int64_t>
getTypeSizeInBytes(mlir::Type type, const mlir::DataLayout &dataLayout) {
  if (!cir::isSized(type))
    return std::nullopt;

  llvm::TypeSize size = dataLayout.getTypeSize(type);
  if (size.isScalable())
    return std::nullopt;
  return size.getFixedValue();
}

/// If \p val is a constant integer that fits in an int64_t, return its value.
/// The constant is interpreted according to the signedness of its type.
static std::optional<int64_t> getConstantIndex(mlir::Value val) {
  auto constOp =
      mlir::dyn_cast_if_present<cir::ConstantOp>(val.getDefiningOp());
  if (!constOp)
    return std::nullopt;

  auto intAttr = mlir::dyn_cast<cir::IntAttr>(constOp.getValue());
  if (!intAttr)
    return std::nullopt;

  const APInt &value = intAttr.getValue();
  if (intAttr.isSigned())
    return value.trySExtValue();
  return value.tryZExtValue();
}

/// Return `count * size`, or std::nullopt if either input is unknown or the
/// product overflows.
static std::optional<int64_t> scaleOffset(std::optional<int64_t> count,
                                          std::optional<int64_t> size) {
  if (!count || !size)
    return std::nullopt;
  auto [product, overflow] = MulOverflow(*count, *size);
  if (overflow)
    return std::nullopt;
  return product;
}

/// Add \p delta bytes to \p offset, making the offset unknown if \p delta is
/// unknown or if the sum overflows.
static void addToOffset(std::optional<int64_t> &offset,
                        std::optional<int64_t> delta) {
  if (!offset)
    return;
  if (!delta) {
    offset.reset();
    return;
  }
  auto [sum, overflow] = AddOverflow(*offset, *delta);
  if (overflow)
    offset.reset();
  else
    offset = sum;
}

namespace {
/// A pointer expressed as a byte offset into the object it points into.
struct PointerOffset {
  /// The value the pointer was traced back to. This is an allocation, a block
  /// argument, or the result of an operation this analysis cannot look through.
  mlir::Value base;

  /// Byte offset of the pointer from the start of `base`. The offset can be
  /// negative. If this is std::nullopt, the offset is not a compile-time
  /// constant.
  std::optional<int64_t> offset;
};
} // namespace

/// Trace \p val back to the object it points into, accumulating the byte offset
/// of \p val from the start of that object.
///
/// The walk stops at a block argument, at an allocation, or at any operation
/// whose result cannot be described as an offset from one of its operands. The
/// returned base and offset always describe \p val, even when the walk stops
/// early because the depth limit was reached.
///
/// Operations contributing an offset that isn't a compile-time constant (a
/// dynamic cir.ptr_stride index, for example) are still traced through, leaving
/// the offset unknown. Knowing which object a pointer points into is useful
/// even when the offset within that object is not known.
static PointerOffset decomposePointer(mlir::Value val,
                                      const mlir::DataLayout &dataLayout) {
  LDBG() << "Decomposing pointer: " << val;

  std::optional<int64_t> offset = 0;

  for (unsigned depth = 0; depth < MaxLookupDepth; ++depth) {
    mlir::Operation *defOp = val.getDefiningOp();
    if (!defOp) {
      LDBG() << "No defining operation, stopping";
      break; // Block argument (e.g. function parameter) — stop here.
    }

    // Bitcasts and address-space casts don't change the address, and
    // array_to_ptrdecay produces a pointer to the first element of the array.
    if (auto castOp = mlir::dyn_cast<cir::CastOp>(defOp)) {
      if (castOp.isAllocaPreservingCast() ||
          castOp.getKind() == cir::CastKind::array_to_ptrdecay) {
        LDBG() << "Walking past cast operation";
        val = castOp.getSrc();
        continue;
      }
      LDBG() << "Opaque cast operation, stopping";
      break;
    }

    // A stride moves the pointer by `stride * sizeof(pointee)` bytes.
    if (auto strideOp = mlir::dyn_cast<cir::PtrStrideOp>(defOp)) {
      LDBG() << "Walking past PtrStrideOp";
      addToOffset(offset,
                  scaleOffset(getConstantIndex(strideOp.getStride()),
                              getTypeSizeInBytes(strideOp.getElementType(),
                                                 dataLayout)));
      val = strideOp.getBase();
      continue;
    }

    // A record member sits at a fixed offset given by the record layout.
    if (auto memberOp = mlir::dyn_cast<cir::GetMemberOp>(defOp)) {
      LDBG() << "Walking past GetMemberOp";
      auto recordTy =
          mlir::cast<cir::RecordType>(memberOp.getAddrTy().getPointee());
      std::optional<int64_t> memberOffset;
      if (!recordTy.isIncomplete())
        memberOffset =
            recordTy.getElementOffset(dataLayout, memberOp.getIndex());
      addToOffset(offset, memberOffset);
      val = memberOp.getAddr();
      continue;
    }

    // An array element sits at `index * sizeof(element)` bytes into the array.
    if (auto elementOp = mlir::dyn_cast<cir::GetElementOp>(defOp)) {
      LDBG() << "Walking past GetElementOp";
      addToOffset(offset,
                  scaleOffset(getConstantIndex(elementOp.getIndex()),
                              getTypeSizeInBytes(elementOp.getElementType(),
                                                 dataLayout)));
      val = elementOp.getBase();
      continue;
    }

    // A base class subobject starts the given number of bytes into the derived
    // object. This may return null if the input is null, but accessing memory
    // based on that null pointer would be UB, so we always assume non-null
    // here.
    if (auto baseOp = mlir::dyn_cast<cir::BaseClassAddrOp>(defOp)) {
      LDBG() << "Walking past BaseClassAddrOp";
      addToOffset(offset, baseOp.getOffset().tryZExtValue());
      val = baseOp.getDerivedAddr();
      continue;
    }

    // Conversely, the derived object starts that many bytes before the base
    // subobject, so the offset is applied as a negative adjustment. This may
    // return null if the input is null, but accessing memory based on that null
    // pointer would be UB, so we always assume non-null here.
    if (auto derivedOp = mlir::dyn_cast<cir::DerivedClassAddrOp>(defOp)) {
      LDBG() << "Walking past DerivedClassAddrOp";
      std::optional<int64_t> baseOffset = derivedOp.getOffset().tryZExtValue();
      if (baseOffset)
        baseOffset = -*baseOffset;
      addToOffset(offset, baseOffset);
      val = derivedOp.getBaseAddr();
      continue;
    }

    // The real part of a complex value is at offset zero, the imaginary part
    // right behind it.
    if (auto realOp = mlir::dyn_cast<cir::ComplexRealPtrOp>(defOp)) {
      LDBG() << "Walking past ComplexRealPtrOp";
      val = realOp.getOperand();
      continue;
    }
    if (auto imagOp = mlir::dyn_cast<cir::ComplexImagPtrOp>(defOp)) {
      LDBG() << "Walking past ComplexImagPtrOp";
      auto ptrTy = mlir::cast<cir::PointerType>(imagOp.getOperand().getType());
      auto complexTy = mlir::cast<cir::ComplexType>(ptrTy.getPointee());
      addToOffset(offset,
                  getTypeSizeInBytes(complexTy.getElementType(), dataLayout));
      val = imagOp.getOperand();
      continue;
    }

    LDBG() << "Unhandled operation, stopping";
    break; // Not expressible as an offset from another pointer.
  }

  return {val, offset};
}

/// Return true if \p lhs and \p rhs are provably different objects.
///
/// TODO: Extend to cover global addresses, function arguments with noalias, and
/// heap allocations.
static bool areDistinctObjects(mlir::Value lhs, mlir::Value rhs) {
  // Distinct cir.alloca ops allocate distinct storage.
  return lhs != rhs &&
         mlir::isa_and_nonnull<cir::AllocaOp>(lhs.getDefiningOp()) &&
         mlir::isa_and_nonnull<cir::AllocaOp>(rhs.getDefiningOp());
}

//===----------------------------------------------------------------------===//
// CIRBasicAliasAnalysis
//===----------------------------------------------------------------------===//

mlir::AliasResult CIRBasicAliasAnalysis::alias(mlir::Value lhs,
                                               mlir::Value rhs) {
  LDBG() << "Checking alias between: " << lhs << " and " << rhs;

  if (lhs == rhs) {
    LDBG() << "Trivial alias between identical values";
    return mlir::AliasResult::MustAlias;
  }

  PointerOffset lhsPtr = decomposePointer(lhs, dataLayout);
  PointerOffset rhsPtr = decomposePointer(rhs, dataLayout);

  if (lhsPtr.base != rhsPtr.base) {
    if (areDistinctObjects(lhsPtr.base, rhsPtr.base)) {
      LDBG() << "No alias between pointers into distinct objects";
      return mlir::AliasResult::NoAlias;
    }
    LDBG() << "Unrelated base objects, may alias";
    return mlir::AliasResult::MayAlias;
  }

  // Both pointers point into the same object, so their offsets can be compared
  // directly.
  if (!lhsPtr.offset || !rhsPtr.offset) {
    LDBG() << "Same object at an unknown offset, may alias";
    return mlir::AliasResult::MayAlias;
  }

  // Equal offsets means both pointers start at exactly the same address, which
  // is all MustAlias claims. How many bytes each access touches doesn't matter.
  if (*lhsPtr.offset == *rhsPtr.offset) {
    LDBG() << "Must alias at the same address within the same object";
    return mlir::AliasResult::MustAlias;
  }

  // TODO: Two pointers at different offsets into the same object only overlap
  // if the accesses are large enough to reach one another. Comparing the byte
  // ranges the accesses cover would prove NoAlias or PartialAlias here.
  LDBG() << "Same object at different offsets, may alias";
  return mlir::AliasResult::MayAlias;
}

mlir::ModRefResult CIRBasicAliasAnalysis::getModRef(mlir::Operation *op,
                                                    mlir::Value location) {
  LDBG() << "getModRef: "
         << mlir::OpWithFlags(op, mlir::OpPrintingFlags().skipRegions())
         << " on location " << location;

  auto effects = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(op);
  if (!effects) {
    LDBG() << "No memory effect interface, returning ModAndRef";
    return mlir::ModRefResult::getModAndRef();
  }

  SmallVector<mlir::MemoryEffects::EffectInstance> effectList;
  effects.getEffects(effectList);

  auto classifyEffect = [location, this](
                            const mlir::MemoryEffects::EffectInstance &effect) {
    if (mlir::isa<mlir::MemoryEffects::Allocate>(effect.getEffect())) {
      LDBG() << "Skipping allocate effect";
      return mlir::ModRefResult::getNoModRef();
    }

    mlir::AliasResult aliasResult = mlir::AliasResult::MayAlias;
    if (mlir::Value affectedLocation = effect.getValue()) {
      LDBG() << "    Checking alias between affected location "
             << affectedLocation << " and query location " << location;
      aliasResult = alias(affectedLocation, location);
      LDBG() << "    Alias result: " << aliasResult;
    } else {
      // An effect on a non-addressable resource cannot affect a
      // pointer-based location.
      if (!effect.getResource()->isAddressable()) {
        LDBG() << "    Effect on non-addressable resource '"
               << effect.getResource()->getName() << "', skipping (NoAlias)";
        aliasResult = mlir::AliasResult::NoAlias;
      } else {
        LDBG() << "    No effect value, assuming MayAlias";
      }
    }

    // If the affected location doesn't alias with the query location,
    // ignore this effect.
    if (aliasResult.isNo()) {
      LDBG() << "No alias with affected location";
      return mlir::ModRefResult::getNoModRef();
    }

    // TODO: Consider whether Free should be NoModRef.
    if (mlir::isa<mlir::MemoryEffects::Free>(effect.getEffect())) {
      LDBG() << "Skipping free effect";
      return mlir::ModRefResult::getModAndRef();
    }

    if (mlir::isa<mlir::MemoryEffects::Write>(effect.getEffect())) {
      LDBG() << "Write effect, adding Mod";
      return mlir::ModRefResult::getMod();
    }

    if (mlir::isa<mlir::MemoryEffects::Read>(effect.getEffect())) {
      LDBG() << "Read effect, adding Ref";
      return mlir::ModRefResult::getRef();
    }

    LDBG() << "Unexpected memory effect: " << effect.getEffect();
    return mlir::ModRefResult::getNoModRef();
  };

  return llvm::accumulate(llvm::map_range(effectList, classifyEffect),
                          mlir::ModRefResult::getNoModRef(),
                          [](mlir::ModRefResult lhs, mlir::ModRefResult rhs) {
                            return lhs.merge(rhs);
                          });
}

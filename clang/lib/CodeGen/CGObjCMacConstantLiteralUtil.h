//===-- CodeGen/CGObjCMacConstantLiteralUtil.h - ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This should be used for things that effect the ABI of
// Obj-C constant initializer literals (`-fobjc-constant-literals`) to allow
// future changes without breaking the ABI promises.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_CODEGEN_CGOBJCMACCONSTANTLITERALUTIL_H
#define LLVM_CLANG_LIB_CODEGEN_CGOBJCMACCONSTANTLITERALUTIL_H

#include "CGObjCRuntime.h"
#include "clang/AST/ExprObjC.h"
#include "clang/AST/Type.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/DenseMapInfo.h"
#include <numeric>

namespace clang {
namespace CodeGen {
namespace CGObjCMacConstantLiteralUtil {

class NSConstantNumberMapInfo {

  enum class MapInfoType {
    Empty,
    Tombstone,
    Int,
    Float,
  };

  MapInfoType InfoType;
  CanQualType QType;
  llvm::APSInt Int;
  llvm::APFloat Float;

  /// Default constructor that can create Empty or Tombstone info entries
  explicit NSConstantNumberMapInfo(MapInfoType I = MapInfoType::Empty)
      : InfoType(I), QType(), Int(), Float(0.0) {}

  bool isEmptyOrTombstone() const {
    return InfoType == MapInfoType::Empty || InfoType == MapInfoType::Tombstone;
  }

public:
  NSConstantNumberMapInfo(CanQualType QT, const llvm::APSInt &V)
      : InfoType(MapInfoType::Int), QType(QT), Int(V), Float(0.0) {}
  NSConstantNumberMapInfo(CanQualType QT, const llvm::APFloat &V)
      : InfoType(MapInfoType::Float), QType(QT), Int(), Float(V) {}

  unsigned getHashValue() const {
    assert(!isEmptyOrTombstone() && "Cannot hash empty or tombstone map info!");

    unsigned QTypeHash = llvm::DenseMapInfo<QualType>::getHashValue(
        llvm::DenseMapInfo<QualType>::getTombstoneKey());

    if (InfoType == MapInfoType::Int)
      return llvm::detail::combineHashValue((unsigned)Int.getZExtValue(),
                                            QTypeHash);

    assert(InfoType == MapInfoType::Float);
    return llvm::detail::combineHashValue(
        (unsigned)Float.bitcastToAPInt().getZExtValue(), QTypeHash);
  }

  static inline NSConstantNumberMapInfo getEmptyKey() {
    return NSConstantNumberMapInfo();
  }

  static inline NSConstantNumberMapInfo getTombstoneKey() {
    return NSConstantNumberMapInfo(MapInfoType::Tombstone);
  }

  bool operator==(const NSConstantNumberMapInfo &RHS) const {
    if (InfoType != RHS.InfoType || QType != RHS.QType)
      return false;

    // Handle the empty and tombstone equality
    if (isEmptyOrTombstone())
      return true;

    if (InfoType == MapInfoType::Int)
      return llvm::APSInt::isSameValue(Int, RHS.Int);

    assert(InfoType == MapInfoType::Float);

    // handle -0, NaN, and infinities correctly
    return Float.bitwiseIsEqual(RHS.Float);
  }
};

using std::iota;

class NSDictionaryBuilder {
  SmallVector<std::pair<llvm::Constant *, llvm::Constant *>, 16> Elements;
  uint64_t Opts;

public:
  enum class Options : uint64_t { Sorted = 1 };

  NSDictionaryBuilder(
      const ObjCDictionaryLiteral *E,
      ArrayRef<std::pair<llvm::Constant *, llvm::Constant *>> KeysAndObjects,
      const Options O = Options::Sorted) {
    Opts = static_cast<uint64_t>(O);
    uint64_t const NumElements = KeysAndObjects.size();

    // Reserve the capacity for the sorted keys & values
    Elements.reserve(NumElements);

    // Setup the element indicies 0 ..< NumElements
    SmallVector<size_t, 16> ElementIndicies(NumElements);
    std::iota(ElementIndicies.begin(), ElementIndicies.end(), 0);

    // Now perform the sorts and shift the indicies as needed
    std::stable_sort(
        ElementIndicies.begin(), ElementIndicies.end(),
        [E, O](size_t LI, size_t RI) {
          Expr *const LK = E->getKeyValueElement(LI).Key->IgnoreImpCasts();
          Expr *const RK = E->getKeyValueElement(RI).Key->IgnoreImpCasts();

          if (!isa<ObjCStringLiteral>(LK) || !isa<ObjCStringLiteral>(RK))
            llvm_unreachable("Non-constant literals should not be sorted to "
                             "maintain existing behavior");

          // NOTE: Using the `StringLiteral->getString()` since it checks that
          //       `chars` are 1 byte
          StringRef LKS = cast<ObjCStringLiteral>(LK)->getString()->getString();
          StringRef RKS = cast<ObjCStringLiteral>(RK)->getString()->getString();

          // Do an alpha sort to aid in with de-dupe at link time
          // `O(log n)` worst case lookup at runtime supported by `Foundation`
          if (O == Options::Sorted)
            return LKS < RKS;
          llvm_unreachable("Unexpected `NSDictionaryBuilder::Options given");
        });

    // Finally use the sorted indicies to insert into `Elements`.
    for (auto &Idx : ElementIndicies) {
      Elements.push_back(KeysAndObjects[Idx]);
    }
  }

  SmallVectorImpl<std::pair<llvm::Constant *, llvm::Constant *>> &
  getElements() {
    return Elements;
  }

  Options getOptions() const { return static_cast<Options>(Opts); }

  uint64_t getNumElements() const { return Elements.size(); }
};

} // namespace CGObjCMacConstantLiteralUtil
} // namespace CodeGen
} // namespace clang

namespace llvm {

using namespace clang::CodeGen::CGObjCMacConstantLiteralUtil;

template <> struct DenseMapInfo<NSConstantNumberMapInfo> {
  static NSConstantNumberMapInfo getEmptyKey() {
    return NSConstantNumberMapInfo::getEmptyKey();
  }

  static NSConstantNumberMapInfo getTombstoneKey() {
    return NSConstantNumberMapInfo::getTombstoneKey();
  }

  static unsigned getHashValue(const NSConstantNumberMapInfo &S) {
    return S.getHashValue();
  }

  static bool isEqual(const NSConstantNumberMapInfo &LHS,
                      const NSConstantNumberMapInfo &RHS) {
    return LHS == RHS;
  }
};

} // namespace llvm

#endif

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
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ConvertUTF.h"
#include <numeric>

namespace clang {
namespace CodeGen {
namespace CGObjCMacConstantLiteralUtil {

class NSConstantNumberMapInfo {

  enum class MapInfoType {
    Empty,
    Int,
    Float,
  };

  MapInfoType InfoType;
  CanQualType QType;
  llvm::APSInt Int;
  llvm::APFloat Float;

  /// Default constructor that can create an Empty info entry.
  explicit NSConstantNumberMapInfo(MapInfoType I = MapInfoType::Empty)
      : InfoType(I), QType(), Int(), Float(0.0) {}

  bool isEmpty() const { return InfoType == MapInfoType::Empty; }

public:
  NSConstantNumberMapInfo(CanQualType QT, const llvm::APSInt &V)
      : InfoType(MapInfoType::Int), QType(QT), Int(V), Float(0.0) {}
  NSConstantNumberMapInfo(CanQualType QT, const llvm::APFloat &V)
      : InfoType(MapInfoType::Float), QType(QT), Int(), Float(V) {}

  unsigned getHashValue() const {
    assert(!isEmpty() && "Cannot hash empty map info!");

    unsigned QTypeHash = llvm::DenseMapInfo<QualType>::getHashValue(QType);

    if (InfoType == MapInfoType::Int)
      return llvm::detail::combineHashValue((unsigned)Int.getZExtValue(),
                                            QTypeHash);

    assert(InfoType == MapInfoType::Float);
    return llvm::detail::combineHashValue(
        (unsigned)Float.bitcastToAPInt().getZExtValue(), QTypeHash);
  }

  bool operator==(const NSConstantNumberMapInfo &RHS) const {
    if (InfoType != RHS.InfoType || QType != RHS.QType)
      return false;

    // Handle the empty equality.
    if (isEmpty())
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

    // Precompute the UTF-16 form of each string key. The runtime stores keys as
    // UTF-16 and looks them up by UTF-16 code-unit order, so we must sort by the
    // same order here. Sorting by the raw UTF-8 bytes instead would diverge for
    // keys mixing characters in U+E000..U+FFFF with astral characters
    // (U+10000..U+10FFFF), because UTF-16 encodes the latter with lead
    // surrogates (0xD800..0xDBFF) that sort *below* 0xE000 -- causing the
    // runtime's lookup to miss keys that are actually present.
    //
    // A key need not be well-formed UTF-8: string literals with invalid or
    // partial sequences are legal in the AST (Sema warns about them separately;
    // see warn_objc_dictionary_ill_formed_utf8_key). We deliberately mirror the
    // constant CFString emitter (CodeGenModule::GetConstantCFStringEntry), which
    // runs ConvertUTF8toUTF16 with strictConversion and keeps whatever prefix
    // converts successfully, so we sort by exactly the code units the string is
    // stored and looked up as. Any trailing bytes that fail to convert are
    // dropped from the sort key; this cannot crash and yields a valid
    // strict-weak ordering regardless of well-formedness.
    SmallVector<SmallVector<llvm::UTF16, 16>, 16> KeysUTF16(NumElements);
    for (size_t I = 0; I < NumElements; ++I) {
      Expr *const K = E->getKeyValueElement(I).Key->IgnoreImpCasts();
      auto *SL = dyn_cast<ObjCStringLiteral>(K);
      assert(SL && "Non-constant literals should not be sorted to "
                   "maintain existing behavior");
      // NOTE: Using the `StringLiteral->getString()` since it checks that
      //       `chars` are 1 byte
      StringRef KS = SL->getString()->getString();
      SmallVectorImpl<llvm::UTF16> &Dst = KeysUTF16[I];
      Dst.resize(KS.size()); // UTF-16 needs <= as many code units as UTF-8.
      const llvm::UTF8 *SrcPtr = reinterpret_cast<const llvm::UTF8 *>(KS.data());
      llvm::UTF16 *DstPtr = Dst.data();
      llvm::ConvertUTF8toUTF16(&SrcPtr, SrcPtr + KS.size(), &DstPtr,
                               DstPtr + Dst.size(), llvm::strictConversion);
      // ConvertUTF8toUTF16 advances DstPtr to the end of the converted prefix.
      Dst.truncate(DstPtr - Dst.data());
    }

    // Now perform the sorts and shift the indicies as needed
    llvm::stable_sort(
        ElementIndicies, [O, &KeysUTF16](size_t LI, size_t RI) {
          // Sort by UTF-16 code unit to match the runtime's lookup order. This
          // is a deterministic total order, so it still aids link-time de-dupe,
          // and it supports the runtime's `O(log n)` worst-case lookup.
          // ArrayRef::operator< is a lexicographic code-unit comparison.
          if (O == Options::Sorted)
            return ArrayRef<llvm::UTF16>(KeysUTF16[LI]) <
                   ArrayRef<llvm::UTF16>(KeysUTF16[RI]);
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

//===- Sanitizers.h - C Language Family Language Options --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Defines the clang::SanitizerKind enum.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_BASIC_SANITIZERS_H
#define LLVM_CLANG_BASIC_SANITIZERS_H

#include "clang/Basic/LLVM.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MathExtras.h"
#include <cassert>
#include <cstdint>

namespace llvm {
class hash_code;
}

namespace clang {

class SanitizerMask {
  /// Number of array elements.
  static constexpr unsigned kNumElem = 2;
  /// Mask value initialized to 0.
  uint64_t maskLoToHigh[kNumElem]{};
  /// Number of bits in a mask.
  static constexpr unsigned kNumBits = sizeof(decltype(maskLoToHigh)) * 8;
  /// Number of bits in a mask element.
  static constexpr unsigned kNumBitElem = sizeof(decltype(maskLoToHigh[0])) * 8;

public:
  static constexpr bool checkBitPos(const unsigned Pos) {
    return Pos < kNumBits;
  }

  /// Create a mask with a bit enabled at position Pos.
  static SanitizerMask bitPosToMask(const unsigned Pos) {
    assert(Pos < kNumBits && "Bit position too big.");
    SanitizerMask mask;
    mask.maskLoToHigh[Pos / kNumBitElem] = 1ULL << Pos % kNumBitElem;
    return mask;
  }

  unsigned countPopulation() const {
    unsigned total = 0;
    for (const auto &Val : maskLoToHigh)
      total += llvm::countPopulation(Val);
    return total;
  }

  void flipAllBits() {
    for (auto &Val : maskLoToHigh)
      Val = ~Val;
  }

  bool isPowerOf2() const {
    return countPopulation() == 1;
  }

  llvm::hash_code hash_value() const;

  explicit operator bool() const {
    for (const auto &Val : maskLoToHigh)
      if (Val)
        return true;
    return false;
  };

  bool operator==(const SanitizerMask &V) const {
    for (unsigned k = 0; k < kNumElem; k++) {
      if (maskLoToHigh[k] != V.maskLoToHigh[k])
        return false;
    }
    return true;
  }

  SanitizerMask &operator&=(const SanitizerMask &RHS) {
    for (unsigned k = 0; k < kNumElem; k++)
      maskLoToHigh[k] &= RHS.maskLoToHigh[k];
    return *this;
  }

  SanitizerMask &operator|=(const SanitizerMask &RHS) {
    for (unsigned k = 0; k < kNumElem; k++)
      maskLoToHigh[k] |= RHS.maskLoToHigh[k];
    return *this;
  }

  bool operator!() const {
    for (const auto &Val : maskLoToHigh)
      if (Val)
        return false;
    return true;
  }

  bool operator!=(const SanitizerMask &RHS) const { return !((*this) == RHS); }
};

// Declaring in clang namespace so that it can be found by ADL.
llvm::hash_code hash_value(const clang::SanitizerMask &Arg);

inline SanitizerMask operator~(SanitizerMask v) {
  v.flipAllBits();
  return v;
}

inline SanitizerMask operator&(SanitizerMask a, const SanitizerMask &b) {
  a &= b;
  return a;
}

inline SanitizerMask operator|(SanitizerMask a, const SanitizerMask &b) {
  a |= b;
  return a;
}

// Define the set of sanitizer kinds, as well as the set of sanitizers each
// sanitizer group expands into.
// Uses static data member of a class template as recommended in second
// workaround from n4424 to avoid odr issues.
// FIXME: Can be marked as constexpr once c++14 can be used in llvm.
// FIXME: n4424 workaround can be replaced by c++17 inline variable.
template <typename T = void> struct SanitizerMasks {

  // Assign ordinals to possible values of -fsanitize= flag, which we will use
  // as bit positions.
  enum SanitizerOrdinal : uint64_t {
#define SANITIZER(NAME, ID) SO_##ID,
#define SANITIZER_GROUP(NAME, ID, ALIAS) SO_##ID##Group,
#include "clang/Basic/Sanitizers.def"
    SO_Count
  };

#define SANITIZER(NAME, ID)                                                    \
  static const SanitizerMask ID;                                               \
  static_assert(SanitizerMask::checkBitPos(SO_##ID), "Bit position too big.");
#define SANITIZER_GROUP(NAME, ID, ALIAS)                                       \
  static const SanitizerMask ID;                                               \
  static const SanitizerMask ID##Group;                                        \
  static_assert(SanitizerMask::checkBitPos(SO_##ID##Group),                    \
                "Bit position too big.");
#include "clang/Basic/Sanitizers.def"
}; // SanitizerMasks

#define SANITIZER(NAME, ID)                                                    \
  template <typename T>                                                        \
  const SanitizerMask SanitizerMasks<T>::ID =                                  \
      SanitizerMask::bitPosToMask(SO_##ID);
#define SANITIZER_GROUP(NAME, ID, ALIAS)                                       \
  template <typename T>                                                        \
  const SanitizerMask SanitizerMasks<T>::ID = SanitizerMask(ALIAS);            \
  template <typename T>                                                        \
  const SanitizerMask SanitizerMasks<T>::ID##Group =                           \
      SanitizerMask::bitPosToMask(SO_##ID##Group);
#include "clang/Basic/Sanitizers.def"

// Explicit instantiation here to ensure correct initialization order.
template struct SanitizerMasks<>;

using SanitizerKind = SanitizerMasks<>;

struct SanitizerSet {
  /// Check if a certain (single) sanitizer is enabled.
  bool has(SanitizerMask K) const {
    assert(K.isPowerOf2() && "Has to be a single sanitizer.");
    return static_cast<bool>(Mask & K);
  }

  /// Check if one or more sanitizers are enabled.
  bool hasOneOf(SanitizerMask K) const { return static_cast<bool>(Mask & K); }

  /// Enable or disable a certain (single) sanitizer.
  void set(SanitizerMask K, bool Value) {
    assert(K.isPowerOf2() && "Has to be a single sanitizer.");
    Mask = Value ? (Mask | K) : (Mask & ~K);
  }

  /// Disable the sanitizers specified in \p K.
  void clear(SanitizerMask K = SanitizerKind::All) { Mask &= ~K; }

  /// Returns true if no sanitizers are enabled.
  bool empty() const { return !Mask; }

  /// Bitmask of enabled sanitizers.
  SanitizerMask Mask;
};

/// Parse a single value from a -fsanitize= or -fno-sanitize= value list.
/// Returns a non-zero SanitizerMask, or \c 0 if \p Value is not known.
SanitizerMask parseSanitizerValue(StringRef Value, bool AllowGroups);

/// For each sanitizer group bit set in \p Kinds, set the bits for sanitizers
/// this group enables.
SanitizerMask expandSanitizerGroups(SanitizerMask Kinds);

/// Return the sanitizers which do not affect preprocessing.
inline SanitizerMask getPPTransparentSanitizers() {
  return SanitizerKind::CFI | SanitizerKind::Integer |
         SanitizerKind::ImplicitConversion | SanitizerKind::Nullability |
         SanitizerKind::Undefined;
}

} // namespace clang

#endif // LLVM_CLANG_BASIC_SANITIZERS_H

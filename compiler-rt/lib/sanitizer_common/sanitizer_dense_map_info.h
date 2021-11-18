//===- sanitizer_dense_map_info.h - Type traits for DenseMap ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SANITIZER_DENSE_MAP_INFO_H
#define SANITIZER_DENSE_MAP_INFO_H

#include "sanitizer_common.h"
#include "sanitizer_internal_defs.h"
#include "sanitizer_type_traits.h"

namespace __sanitizer {

namespace detail {

/// Simplistic combination of 32-bit hash values into 32-bit hash values.
static inline unsigned combineHashValue(unsigned a, unsigned b) {
  u64 key = (u64)a << 32 | (u64)b;
  key += ~(key << 32);
  key ^= (key >> 22);
  key += ~(key << 13);
  key ^= (key >> 8);
  key += (key << 3);
  key ^= (key >> 15);
  key += ~(key << 27);
  key ^= (key >> 31);
  return (unsigned)key;
}

// We extend a pair to allow users to override the bucket type with their own
// implementation without requiring two members.
template <typename KeyT, typename ValueT>
struct DenseMapPair {
  KeyT first = {};
  ValueT second = {};
  DenseMapPair() = default;
  DenseMapPair(const KeyT &f, const ValueT &s) : first(f), second(s) {}

  template <typename KeyT2, typename ValueT2>
  DenseMapPair(KeyT2 &&f, ValueT2 &&s)
      : first(__sanitizer::forward<KeyT2>(f)),
        second(__sanitizer::forward<ValueT2>(s)) {}

  DenseMapPair(const DenseMapPair &other) = default;
  DenseMapPair &operator=(const DenseMapPair &other) = default;
  DenseMapPair(DenseMapPair &&other) = default;
  DenseMapPair &operator=(DenseMapPair &&other) = default;

  KeyT &getFirst() { return first; }
  const KeyT &getFirst() const { return first; }
  ValueT &getSecond() { return second; }
  const ValueT &getSecond() const { return second; }
};

}  // end namespace detail

template <typename T>
struct DenseMapInfo {
  // static inline T getEmptyKey();
  // static inline T getTombstoneKey();
  // static unsigned getHashValue(const T &Val);
  // static bool isEqual(const T &LHS, const T &RHS);
};

// Provide DenseMapInfo for all pointers. Come up with sentinel pointer values
// that are aligned to alignof(T) bytes, but try to avoid requiring T to be
// complete. This allows clients to instantiate DenseMap<T*, ...> with forward
// declared key types. Assume that no pointer key type requires more than 4096
// bytes of alignment.
template <typename T>
struct DenseMapInfo<T *> {
  // The following should hold, but it would require T to be complete:
  // static_assert(alignof(T) <= (1 << Log2MaxAlign),
  //               "DenseMap does not support pointer keys requiring more than "
  //               "Log2MaxAlign bits of alignment");
  static constexpr uptr Log2MaxAlign = 12;

  static inline T *getEmptyKey() {
    uptr Val = static_cast<uptr>(-1);
    Val <<= Log2MaxAlign;
    return reinterpret_cast<T *>(Val);
  }

  static inline T *getTombstoneKey() {
    uptr Val = static_cast<uptr>(-2);
    Val <<= Log2MaxAlign;
    return reinterpret_cast<T *>(Val);
  }

  static unsigned getHashValue(const T *PtrVal) {
    return (unsigned((uptr)PtrVal) >> 4) ^ (unsigned((uptr)PtrVal) >> 9);
  }

  static bool isEqual(const T *LHS, const T *RHS) { return LHS == RHS; }
};

// Provide DenseMapInfo for chars.
template <>
struct DenseMapInfo<char> {
  static inline char getEmptyKey() { return ~0; }
  static inline char getTombstoneKey() { return ~0 - 1; }
  static unsigned getHashValue(const char &Val) { return Val * 37U; }

  static bool isEqual(const char &LHS, const char &RHS) { return LHS == RHS; }
};

// Provide DenseMapInfo for unsigned chars.
template <>
struct DenseMapInfo<unsigned char> {
  static inline unsigned char getEmptyKey() { return ~0; }
  static inline unsigned char getTombstoneKey() { return ~0 - 1; }
  static unsigned getHashValue(const unsigned char &Val) { return Val * 37U; }

  static bool isEqual(const unsigned char &LHS, const unsigned char &RHS) {
    return LHS == RHS;
  }
};

// Provide DenseMapInfo for unsigned shorts.
template <>
struct DenseMapInfo<unsigned short> {
  static inline unsigned short getEmptyKey() { return 0xFFFF; }
  static inline unsigned short getTombstoneKey() { return 0xFFFF - 1; }
  static unsigned getHashValue(const unsigned short &Val) { return Val * 37U; }

  static bool isEqual(const unsigned short &LHS, const unsigned short &RHS) {
    return LHS == RHS;
  }
};

// Provide DenseMapInfo for unsigned ints.
template <>
struct DenseMapInfo<unsigned> {
  static inline unsigned getEmptyKey() { return ~0U; }
  static inline unsigned getTombstoneKey() { return ~0U - 1; }
  static unsigned getHashValue(const unsigned &Val) { return Val * 37U; }

  static bool isEqual(const unsigned &LHS, const unsigned &RHS) {
    return LHS == RHS;
  }
};

// Provide DenseMapInfo for unsigned longs.
template <>
struct DenseMapInfo<unsigned long> {
  static inline unsigned long getEmptyKey() { return ~0UL; }
  static inline unsigned long getTombstoneKey() { return ~0UL - 1L; }

  static unsigned getHashValue(const unsigned long &Val) {
    return (unsigned)(Val * 37UL);
  }

  static bool isEqual(const unsigned long &LHS, const unsigned long &RHS) {
    return LHS == RHS;
  }
};

// Provide DenseMapInfo for unsigned long longs.
template <>
struct DenseMapInfo<unsigned long long> {
  static inline unsigned long long getEmptyKey() { return ~0ULL; }
  static inline unsigned long long getTombstoneKey() { return ~0ULL - 1ULL; }

  static unsigned getHashValue(const unsigned long long &Val) {
    return (unsigned)(Val * 37ULL);
  }

  static bool isEqual(const unsigned long long &LHS,
                      const unsigned long long &RHS) {
    return LHS == RHS;
  }
};

// Provide DenseMapInfo for shorts.
template <>
struct DenseMapInfo<short> {
  static inline short getEmptyKey() { return 0x7FFF; }
  static inline short getTombstoneKey() { return -0x7FFF - 1; }
  static unsigned getHashValue(const short &Val) { return Val * 37U; }
  static bool isEqual(const short &LHS, const short &RHS) { return LHS == RHS; }
};

// Provide DenseMapInfo for ints.
template <>
struct DenseMapInfo<int> {
  static inline int getEmptyKey() { return 0x7fffffff; }
  static inline int getTombstoneKey() { return -0x7fffffff - 1; }
  static unsigned getHashValue(const int &Val) { return (unsigned)(Val * 37U); }

  static bool isEqual(const int &LHS, const int &RHS) { return LHS == RHS; }
};

// Provide DenseMapInfo for longs.
template <>
struct DenseMapInfo<long> {
  static inline long getEmptyKey() {
    return (1UL << (sizeof(long) * 8 - 1)) - 1UL;
  }

  static inline long getTombstoneKey() { return getEmptyKey() - 1L; }

  static unsigned getHashValue(const long &Val) {
    return (unsigned)(Val * 37UL);
  }

  static bool isEqual(const long &LHS, const long &RHS) { return LHS == RHS; }
};

// Provide DenseMapInfo for long longs.
template <>
struct DenseMapInfo<long long> {
  static inline long long getEmptyKey() { return 0x7fffffffffffffffLL; }
  static inline long long getTombstoneKey() {
    return -0x7fffffffffffffffLL - 1;
  }

  static unsigned getHashValue(const long long &Val) {
    return (unsigned)(Val * 37ULL);
  }

  static bool isEqual(const long long &LHS, const long long &RHS) {
    return LHS == RHS;
  }
};

// Provide DenseMapInfo for all pairs whose members have info.
template <typename T, typename U>
struct DenseMapInfo<detail::DenseMapPair<T, U>> {
  using Pair = detail::DenseMapPair<T, U>;
  using FirstInfo = DenseMapInfo<T>;
  using SecondInfo = DenseMapInfo<U>;

  static inline Pair getEmptyKey() {
    return detail::DenseMapPair<T, U>(FirstInfo::getEmptyKey(),
                                      SecondInfo::getEmptyKey());
  }

  static inline Pair getTombstoneKey() {
    return detail::DenseMapPair<T, U>(FirstInfo::getTombstoneKey(),
                                      SecondInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const Pair &PairVal) {
    return detail::combineHashValue(FirstInfo::getHashValue(PairVal.first),
                                    SecondInfo::getHashValue(PairVal.second));
  }

  static bool isEqual(const Pair &LHS, const Pair &RHS) {
    return FirstInfo::isEqual(LHS.first, RHS.first) &&
           SecondInfo::isEqual(LHS.second, RHS.second);
  }
};

}  // namespace __sanitizer

#endif  // SANITIZER_DENSE_MAP_INFO_H

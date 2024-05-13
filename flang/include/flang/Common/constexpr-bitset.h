//===-- include/flang/Common/constexpr-bitset.h -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_COMMON_CONSTEXPR_BITSET_H_
#define FORTRAN_COMMON_CONSTEXPR_BITSET_H_

// Implements a replacement for std::bitset<> that is suitable for use
// in constexpr expressions.  Limited to elements in [0..127].

#include "bit-population-count.h"
#include "uint128.h"
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <optional>
#include <type_traits>

namespace Fortran::common {

template <int BITS> class BitSet {
  static_assert(BITS > 0 && BITS <= 128);
  using Word = HostUnsignedIntType<(BITS <= 32 ? 32 : BITS)>;
  static constexpr Word allBits{
      ~static_cast<Word>(0) >> (8 * sizeof(Word) - BITS)};

  constexpr BitSet(Word b) : bits_{b} {}

public:
  constexpr BitSet() {}
  constexpr BitSet(const std::initializer_list<int> &xs) {
    for (auto x : xs) {
      set(x);
    }
  }
  constexpr BitSet(const BitSet &) = default;
  constexpr BitSet(BitSet &&) = default;
  constexpr BitSet &operator=(const BitSet &) = default;
  constexpr BitSet &operator=(BitSet &&) = default;

  constexpr BitSet &operator&=(const BitSet &that) {
    bits_ &= that.bits_;
    return *this;
  }
  constexpr BitSet &operator&=(BitSet &&that) {
    bits_ &= that.bits_;
    return *this;
  }
  constexpr BitSet &operator^=(const BitSet &that) {
    bits_ ^= that.bits_;
    return *this;
  }
  constexpr BitSet &operator^=(BitSet &&that) {
    bits_ ^= that.bits_;
    return *this;
  }
  constexpr BitSet &operator|=(const BitSet &that) {
    bits_ |= that.bits_;
    return *this;
  }
  constexpr BitSet &operator|=(BitSet &&that) {
    bits_ |= that.bits_;
    return *this;
  }

  constexpr BitSet operator~() const { return ~bits_; }
  constexpr BitSet operator&(const BitSet &that) const {
    return bits_ & that.bits_;
  }
  constexpr BitSet operator&(BitSet &&that) const { return bits_ & that.bits_; }
  constexpr BitSet operator^(const BitSet &that) const {
    return bits_ ^ that.bits_;
  }
  constexpr BitSet operator^(BitSet &&that) const { return bits_ & that.bits_; }
  constexpr BitSet operator|(const BitSet &that) const {
    return bits_ | that.bits_;
  }
  constexpr BitSet operator|(BitSet &&that) const { return bits_ | that.bits_; }

  constexpr bool operator==(const BitSet &that) const {
    return bits_ == that.bits_;
  }
  constexpr bool operator==(BitSet &&that) const { return bits_ == that.bits_; }
  constexpr bool operator!=(const BitSet &that) const {
    return bits_ != that.bits_;
  }
  constexpr bool operator!=(BitSet &&that) const { return bits_ != that.bits_; }

  static constexpr std::size_t size() { return BITS; }
  constexpr bool test(std::size_t x) const {
    return x < BITS && ((bits_ >> x) & 1) != 0;
  }

  constexpr bool all() const { return bits_ == allBits; }
  constexpr bool any() const { return bits_ != 0; }
  constexpr bool none() const { return bits_ == 0; }

  constexpr std::size_t count() const { return BitPopulationCount(bits_); }

  constexpr BitSet &set() {
    bits_ = allBits;
    return *this;
  }
  constexpr BitSet set(std::size_t x, bool value = true) {
    if (!value) {
      return reset(x);
    } else {
      bits_ |= static_cast<Word>(1) << x;
      return *this;
    }
  }
  constexpr BitSet &reset() {
    bits_ = 0;
    return *this;
  }
  constexpr BitSet &reset(std::size_t x) {
    bits_ &= ~(static_cast<Word>(1) << x);
    return *this;
  }
  constexpr BitSet &flip() {
    bits_ ^= allBits;
    return *this;
  }
  constexpr BitSet &flip(std::size_t x) {
    bits_ ^= static_cast<Word>(1) << x;
    return *this;
  }

  constexpr std::optional<std::size_t> LeastElement() const {
    if (bits_ == 0) {
      return std::nullopt;
    } else {
      return {TrailingZeroBitCount(bits_)};
    }
  }

  Word bits() const { return bits_; }

private:
  Word bits_{0};
};
} // namespace Fortran::common
#endif // FORTRAN_COMMON_CONSTEXPR_BITSET_H_

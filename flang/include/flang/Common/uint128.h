//===-- include/flang/Common/uint128.h --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Portable 128-bit unsigned integer arithmetic for use in impoverished
// C++ implementations lacking __uint128_t.

#ifndef FORTRAN_COMMON_UINT128_H_
#define FORTRAN_COMMON_UINT128_H_

// Define AVOID_NATIVE_UINT128_T to force the use of UnsignedInt128 below
// instead of the C++ compiler's native 128-bit unsigned integer type, if
// it has one.
#ifndef AVOID_NATIVE_UINT128_T
#define AVOID_NATIVE_UINT128_T 0
#endif

#include "leading-zero-bit-count.h"
#include <cstdint>
#include <type_traits>

namespace Fortran::common {

class UnsignedInt128 {
public:
  constexpr UnsignedInt128() {}
  // This means of definition provides some portability for
  // "size_t" operands.
  constexpr UnsignedInt128(unsigned n) : low_{n} {}
  constexpr UnsignedInt128(unsigned long n) : low_{n} {}
  constexpr UnsignedInt128(unsigned long long n) : low_{n} {}
  constexpr UnsignedInt128(int n)
      : low_{static_cast<std::uint64_t>(n)}, high_{-static_cast<std::uint64_t>(
                                                 n < 0)} {}
  constexpr UnsignedInt128(long n)
      : low_{static_cast<std::uint64_t>(n)}, high_{-static_cast<std::uint64_t>(
                                                 n < 0)} {}
  constexpr UnsignedInt128(long long n)
      : low_{static_cast<std::uint64_t>(n)}, high_{-static_cast<std::uint64_t>(
                                                 n < 0)} {}
  constexpr UnsignedInt128(const UnsignedInt128 &) = default;
  constexpr UnsignedInt128(UnsignedInt128 &&) = default;
  constexpr UnsignedInt128 &operator=(const UnsignedInt128 &) = default;
  constexpr UnsignedInt128 &operator=(UnsignedInt128 &&) = default;

  constexpr UnsignedInt128 operator+() const { return *this; }
  constexpr UnsignedInt128 operator~() const { return {~high_, ~low_}; }
  constexpr UnsignedInt128 operator-() const { return ~*this + 1; }
  constexpr bool operator!() const { return !low_ && !high_; }
  constexpr explicit operator bool() const { return low_ || high_; }
  constexpr explicit operator std::uint64_t() const { return low_; }
  constexpr explicit operator std::int64_t() const { return low_; }
  constexpr explicit operator int() const { return static_cast<int>(low_); }

  constexpr std::uint64_t high() const { return high_; }
  constexpr std::uint64_t low() const { return low_; }

  constexpr UnsignedInt128 operator++(/*prefix*/) {
    *this += 1;
    return *this;
  }
  constexpr UnsignedInt128 operator++(int /*postfix*/) {
    UnsignedInt128 result{*this};
    *this += 1;
    return result;
  }
  constexpr UnsignedInt128 operator--(/*prefix*/) {
    *this -= 1;
    return *this;
  }
  constexpr UnsignedInt128 operator--(int /*postfix*/) {
    UnsignedInt128 result{*this};
    *this -= 1;
    return result;
  }

  constexpr UnsignedInt128 operator&(UnsignedInt128 that) const {
    return {high_ & that.high_, low_ & that.low_};
  }
  constexpr UnsignedInt128 operator|(UnsignedInt128 that) const {
    return {high_ | that.high_, low_ | that.low_};
  }
  constexpr UnsignedInt128 operator^(UnsignedInt128 that) const {
    return {high_ ^ that.high_, low_ ^ that.low_};
  }

  constexpr UnsignedInt128 operator<<(UnsignedInt128 that) const {
    if (that >= 128) {
      return {};
    } else if (that == 0) {
      return *this;
    } else {
      std::uint64_t n{that.low_};
      if (n >= 64) {
        return {low_ << (n - 64), 0};
      } else {
        return {(high_ << n) | (low_ >> (64 - n)), low_ << n};
      }
    }
  }
  constexpr UnsignedInt128 operator>>(UnsignedInt128 that) const {
    if (that >= 128) {
      return {};
    } else if (that == 0) {
      return *this;
    } else {
      std::uint64_t n{that.low_};
      if (n >= 64) {
        return {0, high_ >> (n - 64)};
      } else {
        return {high_ >> n, (high_ << (64 - n)) | (low_ >> n)};
      }
    }
  }

  constexpr UnsignedInt128 operator+(UnsignedInt128 that) const {
    std::uint64_t lower{(low_ & ~topBit) + (that.low_ & ~topBit)};
    bool carry{((lower >> 63) + (low_ >> 63) + (that.low_ >> 63)) > 1};
    return {high_ + that.high_ + carry, low_ + that.low_};
  }
  constexpr UnsignedInt128 operator-(UnsignedInt128 that) const {
    return *this + -that;
  }

  constexpr UnsignedInt128 operator*(UnsignedInt128 that) const {
    std::uint64_t mask32{0xffffffff};
    if (high_ == 0 && that.high_ == 0) {
      std::uint64_t x0{low_ & mask32}, x1{low_ >> 32};
      std::uint64_t y0{that.low_ & mask32}, y1{that.low_ >> 32};
      UnsignedInt128 x0y0{x0 * y0}, x0y1{x0 * y1};
      UnsignedInt128 x1y0{x1 * y0}, x1y1{x1 * y1};
      return x0y0 + ((x0y1 + x1y0) << 32) + (x1y1 << 64);
    } else {
      std::uint64_t x0{low_ & mask32}, x1{low_ >> 32}, x2{high_ & mask32},
          x3{high_ >> 32};
      std::uint64_t y0{that.low_ & mask32}, y1{that.low_ >> 32},
          y2{that.high_ & mask32}, y3{that.high_ >> 32};
      UnsignedInt128 x0y0{x0 * y0}, x0y1{x0 * y1}, x0y2{x0 * y2}, x0y3{x0 * y3};
      UnsignedInt128 x1y0{x1 * y0}, x1y1{x1 * y1}, x1y2{x1 * y2};
      UnsignedInt128 x2y0{x2 * y0}, x2y1{x2 * y1};
      UnsignedInt128 x3y0{x3 * y0};
      return x0y0 + ((x0y1 + x1y0) << 32) + ((x0y2 + x1y1 + x2y0) << 64) +
          ((x0y3 + x1y2 + x2y1 + x3y0) << 96);
    }
  }

  constexpr UnsignedInt128 operator/(UnsignedInt128 that) const {
    int j{LeadingZeroes()};
    UnsignedInt128 bits{*this};
    bits <<= j;
    UnsignedInt128 numerator{};
    UnsignedInt128 quotient{};
    for (; j < 128; ++j) {
      numerator <<= 1;
      if (bits.high_ & topBit) {
        numerator.low_ |= 1;
      }
      bits <<= 1;
      quotient <<= 1;
      if (numerator >= that) {
        ++quotient;
        numerator -= that;
      }
    }
    return quotient;
  }

  constexpr UnsignedInt128 operator%(UnsignedInt128 that) const {
    int j{LeadingZeroes()};
    UnsignedInt128 bits{*this};
    bits <<= j;
    UnsignedInt128 remainder{};
    for (; j < 128; ++j) {
      remainder <<= 1;
      if (bits.high_ & topBit) {
        remainder.low_ |= 1;
      }
      bits <<= 1;
      if (remainder >= that) {
        remainder -= that;
      }
    }
    return remainder;
  }

  constexpr bool operator<(UnsignedInt128 that) const {
    return high_ < that.high_ || (high_ == that.high_ && low_ < that.low_);
  }
  constexpr bool operator<=(UnsignedInt128 that) const {
    return !(*this > that);
  }
  constexpr bool operator==(UnsignedInt128 that) const {
    return low_ == that.low_ && high_ == that.high_;
  }
  constexpr bool operator!=(UnsignedInt128 that) const {
    return !(*this == that);
  }
  constexpr bool operator>=(UnsignedInt128 that) const { return that <= *this; }
  constexpr bool operator>(UnsignedInt128 that) const { return that < *this; }

  constexpr UnsignedInt128 &operator&=(const UnsignedInt128 &that) {
    *this = *this & that;
    return *this;
  }
  constexpr UnsignedInt128 &operator|=(const UnsignedInt128 &that) {
    *this = *this | that;
    return *this;
  }
  constexpr UnsignedInt128 &operator^=(const UnsignedInt128 &that) {
    *this = *this ^ that;
    return *this;
  }
  constexpr UnsignedInt128 &operator<<=(const UnsignedInt128 &that) {
    *this = *this << that;
    return *this;
  }
  constexpr UnsignedInt128 &operator>>=(const UnsignedInt128 &that) {
    *this = *this >> that;
    return *this;
  }
  constexpr UnsignedInt128 &operator+=(const UnsignedInt128 &that) {
    *this = *this + that;
    return *this;
  }
  constexpr UnsignedInt128 &operator-=(const UnsignedInt128 &that) {
    *this = *this - that;
    return *this;
  }
  constexpr UnsignedInt128 &operator*=(const UnsignedInt128 &that) {
    *this = *this * that;
    return *this;
  }
  constexpr UnsignedInt128 &operator/=(const UnsignedInt128 &that) {
    *this = *this / that;
    return *this;
  }
  constexpr UnsignedInt128 &operator%=(const UnsignedInt128 &that) {
    *this = *this % that;
    return *this;
  }

private:
  constexpr UnsignedInt128(std::uint64_t hi, std::uint64_t lo)
      : low_{lo}, high_{hi} {}
  constexpr int LeadingZeroes() const {
    if (high_ == 0) {
      return 64 + LeadingZeroBitCount(low_);
    } else {
      return LeadingZeroBitCount(high_);
    }
  }
  static constexpr std::uint64_t topBit{std::uint64_t{1} << 63};
  std::uint64_t low_{0}, high_{0};
};

#if AVOID_NATIVE_UINT128_T
using uint128_t = UnsignedInt128;
#elif (defined __GNUC__ || defined __clang__) && defined __SIZEOF_INT128__
using uint128_t = __uint128_t;
#else
using uint128_t = UnsignedInt128;
#endif

template <int BITS> struct HostUnsignedIntTypeHelper {
  using type = std::conditional_t<(BITS <= 8), std::uint8_t,
      std::conditional_t<(BITS <= 16), std::uint16_t,
          std::conditional_t<(BITS <= 32), std::uint32_t,
              std::conditional_t<(BITS <= 64), std::uint64_t, uint128_t>>>>;
};
template <int BITS>
using HostUnsignedIntType = typename HostUnsignedIntTypeHelper<BITS>::type;

} // namespace Fortran::common
#endif

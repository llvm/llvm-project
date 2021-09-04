//===-- A class to manipulate wide integers. --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_UTILS_FPUTIL_UINT_H
#define LLVM_LIBC_UTILS_FPUTIL_UINT_H

#include <stddef.h> // For size_t
#include <stdint.h>

namespace __llvm_libc {
namespace fputil {

template <size_t Bits> class UInt {

  // This is mainly used for debugging.
  enum Kind {
    NotANumber,
    Valid,
  };

  static_assert(Bits > 0 && Bits % 64 == 0,
                "Number of bits in UInt should be a multiple of 64.");
  static constexpr uint64_t Mask32 = 0xFFFFFFFF;
  static constexpr size_t WordCount = Bits / 64;
  static constexpr uint64_t InvalidHexDigit = 20;
  uint64_t val[WordCount];
  Kind kind;

  uint64_t low(uint64_t v) { return v & Mask32; }

  uint64_t high(uint64_t v) { return (v >> 32) & Mask32; }

  uint64_t hexval(char c) {
    uint64_t diff;
    if ((diff = uint64_t(c) - 'A') < 6)
      return diff + 10;
    else if ((diff = uint64_t(c) - 'a') < 6)
      return diff + 10;
    else if ((diff = uint64_t(c) - '0') < 10)
      return diff;
    else
      return InvalidHexDigit;
  }

  size_t strlen(const char *s) {
    size_t len;
    for (len = 0; *s != '\0'; ++s, ++len)
      ;
    return len;
  }

public:
  UInt() { kind = Valid; }

  UInt(const UInt<Bits> &other) : kind(other.kind) {
    if (kind == Valid) {
      for (size_t i = 0; i < WordCount; ++i)
        val[i] = other.val[i];
    }
  }

  // This constructor is used for debugging.
  explicit UInt(const char *s) {
    size_t len = strlen(s);
    if (len > Bits / 4 + 2 || len < 3) {
      kind = NotANumber;
      return;
    }

    if (!(s[0] == '0' && s[1] == 'x')) {
      kind = NotANumber;
      return;
    }

    for (size_t i = 0; i < WordCount; ++i)
      val[i] = 0;

    for (size_t i = len - 1, w = 0; i >= 2; --i, w += 4) {
      uint64_t hex = hexval(s[i]);
      if (hex == InvalidHexDigit) {
        kind = NotANumber;
        return;
      }
      val[w / 64] |= (hex << (w % 64));
    }

    kind = Valid;
  }

  explicit UInt(uint64_t v) {
    val[0] = v;
    for (size_t i = 1; i < WordCount; ++i)
      val[i] = 0;
    kind = Valid;
  }

  explicit UInt(uint64_t data[WordCount]) {
    for (size_t i = 0; i < WordCount; ++i)
      val[i] = data[i];
    kind = Valid;
  }

  bool is_valid() const { return kind == Valid; }

  // Add x to this number and store the result in this number.
  // Returns the carry value produced by the addition operation.
  uint64_t add(const UInt<Bits> &x) {
    uint64_t carry = 0;
    for (size_t i = 0; i < WordCount; ++i) {
      uint64_t res_lo = low(val[i]) + low(x.val[i]) + carry;
      carry = high(res_lo);
      res_lo = low(res_lo);

      uint64_t res_hi = high(val[i]) + high(x.val[i]) + carry;
      carry = high(res_hi);
      res_hi = low(res_hi);

      val[i] = res_lo + (res_hi << 32);
    }
    return carry;
  }

  // Multiply this number with x and store the result in this number. It is
  // implemented using the long multiplication algorithm by splitting the
  // 64-bit words of this number and |x| in to 32-bit halves but peforming
  // the operations using 64-bit numbers. This ensures that we don't lose the
  // carry bits.
  // Returns the carry value produced by the multiplication operation.
  uint64_t mul(uint64_t x) {
    uint64_t x_lo = low(x);
    uint64_t x_hi = high(x);

    uint64_t row1[WordCount + 1];
    uint64_t carry = 0;
    for (size_t i = 0; i < WordCount; ++i) {
      uint64_t l = low(val[i]);
      uint64_t h = high(val[i]);
      uint64_t p1 = x_lo * l;
      uint64_t p2 = x_lo * h;

      uint64_t res_lo = low(p1) + carry;
      carry = high(res_lo);
      uint64_t res_hi = high(p1) + low(p2) + carry;
      carry = high(res_hi) + high(p2);

      res_lo = low(res_lo);
      res_hi = low(res_hi);
      row1[i] = res_lo + (res_hi << 32);
    }
    row1[WordCount] = carry;

    uint64_t row2[WordCount + 1];
    row2[0] = 0;
    carry = 0;
    for (size_t i = 0; i < WordCount; ++i) {
      uint64_t l = low(val[i]);
      uint64_t h = high(val[i]);
      uint64_t p1 = x_hi * l;
      uint64_t p2 = x_hi * h;

      uint64_t res_lo = low(p1) + carry;
      carry = high(res_lo);
      uint64_t res_hi = high(p1) + low(p2) + carry;
      carry = high(res_hi) + high(p2);

      res_lo = low(res_lo);
      res_hi = low(res_hi);
      row2[i] = res_lo + (res_hi << 32);
    }
    row2[WordCount] = carry;

    UInt<(WordCount + 1) * 64> r1(row1), r2(row2);
    r2.shift_left(32);
    r1.add(r2);
    for (size_t i = 0; i < WordCount; ++i) {
      val[i] = r1[i];
    }
    return r1[WordCount];
  }

  void shift_left(size_t s) {
    const size_t drop = s / 64;  // Number of words to drop
    const size_t shift = s % 64; // Bits to shift in the remaining words.
    const uint64_t mask = ((uint64_t(1) << shift) - 1) << (64 - shift);

    for (size_t i = WordCount; drop > 0 && i > 0; --i) {
      if (i - drop > 0)
        val[i - 1] = val[i - drop - 1];
      else
        val[i - 1] = 0;
    }
    for (size_t i = WordCount; shift > 0 && i > drop; --i) {
      uint64_t drop_val = (val[i - 1] & mask) >> (64 - shift);
      val[i - 1] <<= shift;
      if (i < WordCount)
        val[i] |= drop_val;
    }
  }

  void shift_right(size_t s) {
    const size_t drop = s / 64;  // Number of words to drop
    const size_t shift = s % 64; // Bit shift in the remaining words.
    const uint64_t mask = (uint64_t(1) << shift) - 1;

    for (size_t i = 0; drop > 0 && i < WordCount; ++i) {
      if (i + drop < WordCount)
        val[i] = val[i + drop];
      else
        val[i] = 0;
    }
    for (size_t i = 0; shift > 0 && i < WordCount; ++i) {
      uint64_t drop_val = ((val[i] & mask) << (64 - shift));
      val[i] >>= shift;
      if (i > 0)
        val[i - 1] |= drop_val;
    }
  }

  const uint64_t &operator[](size_t i) const { return val[i]; }

  uint64_t &operator[](size_t i) { return val[i]; }

  uint64_t *data() { return val; }

  const uint64_t *data() const { return val; }
};

} // namespace fputil
} // namespace __llvm_libc

#endif // LLVM_LIBC_UTILS_FPUTIL_UINT_H

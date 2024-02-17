//===-- Mimics llvm/Support/MathExtras.h ------------------------*- C++ -*-===//
// Provides useful math functions.
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_MATH_EXTRAS_H
#define LLVM_LIBC_SRC___SUPPORT_MATH_EXTRAS_H

#include "src/__support/CPP/limits.h"        // CHAR_BIT
#include "src/__support/CPP/type_traits.h"   // is_unsigned_v
#include "src/__support/macros/attributes.h" // LIBC_INLINE
#include "src/__support/macros/config.h"     // LIBC_HAS_BUILTIN

namespace LIBC_NAMESPACE {

// Create a bitmask with the count right-most bits set to 1, and all other bits
// set to 0.  Only unsigned types are allowed.
template <typename T, size_t count>
LIBC_INLINE constexpr T mask_trailing_ones() {
  static_assert(cpp::is_unsigned_v<T>);
  constexpr unsigned t_bits = CHAR_BIT * sizeof(T);
  static_assert(count <= t_bits && "Invalid bit index");
  // It's important not to initialize T with -1, since T may be BigInt which
  // will take -1 as a uint64_t and only initialize the low 64 bits.
  constexpr T all_zeroes(0);
  constexpr T all_ones(~all_zeroes); // bitwise NOT performs integer promotion.
  return count == 0 ? 0 : (all_ones >> (t_bits - count));
}

// Create a bitmask with the count left-most bits set to 1, and all other bits
// set to 0.  Only unsigned types are allowed.
template <typename T, size_t count>
LIBC_INLINE constexpr T mask_leading_ones() {
  constexpr T mask(mask_trailing_ones<T, CHAR_BIT * sizeof(T) - count>());
  return T(~mask); // bitwise NOT performs integer promotion.
}

// Add with carry
template <typename T> struct SumCarry {
  T sum;
  T carry;
};

// This version is always valid for constexpr.
template <typename T>
LIBC_INLINE constexpr cpp::enable_if_t<
    cpp::is_integral_v<T> && cpp::is_unsigned_v<T>, SumCarry<T>>
add_with_carry_const(T a, T b, T carry_in) {
  T tmp = a + carry_in;
  T sum = b + tmp;
  T carry_out = (sum < b) + (tmp < a);
  return {sum, carry_out};
}

template <typename T>
LIBC_INLINE constexpr cpp::enable_if_t<
    cpp::is_integral_v<T> && cpp::is_unsigned_v<T>, SumCarry<T>>
add_with_carry(T a, T b, T carry_in) {
  return add_with_carry_const<T>(a, b, carry_in);
}

#if LIBC_HAS_BUILTIN(__builtin_addc)
// https://clang.llvm.org/docs/LanguageExtensions.html#multiprecision-arithmetic-builtins

template <>
LIBC_INLINE constexpr SumCarry<unsigned char>
add_with_carry<unsigned char>(unsigned char a, unsigned char b,
                              unsigned char carry_in) {
  if (__builtin_is_constant_evaluated()) {
    return add_with_carry_const<unsigned char>(a, b, carry_in);
  } else {
    SumCarry<unsigned char> result{0, 0};
    result.sum = __builtin_addcb(a, b, carry_in, &result.carry);
    return result;
  }
}

template <>
LIBC_INLINE constexpr SumCarry<unsigned short>
add_with_carry<unsigned short>(unsigned short a, unsigned short b,
                               unsigned short carry_in) {
  if (__builtin_is_constant_evaluated()) {
    return add_with_carry_const<unsigned short>(a, b, carry_in);
  } else {
    SumCarry<unsigned short> result{0, 0};
    result.sum = __builtin_addcs(a, b, carry_in, &result.carry);
    return result;
  }
}

template <>
LIBC_INLINE constexpr SumCarry<unsigned int>
add_with_carry<unsigned int>(unsigned int a, unsigned int b,
                             unsigned int carry_in) {
  if (__builtin_is_constant_evaluated()) {
    return add_with_carry_const<unsigned int>(a, b, carry_in);
  } else {
    SumCarry<unsigned int> result{0, 0};
    result.sum = __builtin_addc(a, b, carry_in, &result.carry);
    return result;
  }
}

template <>
LIBC_INLINE constexpr SumCarry<unsigned long>
add_with_carry<unsigned long>(unsigned long a, unsigned long b,
                              unsigned long carry_in) {
  if (__builtin_is_constant_evaluated()) {
    return add_with_carry_const<unsigned long>(a, b, carry_in);
  } else {
    SumCarry<unsigned long> result{0, 0};
    result.sum = __builtin_addcl(a, b, carry_in, &result.carry);
    return result;
  }
}

template <>
LIBC_INLINE constexpr SumCarry<unsigned long long>
add_with_carry<unsigned long long>(unsigned long long a, unsigned long long b,
                                   unsigned long long carry_in) {
  if (__builtin_is_constant_evaluated()) {
    return add_with_carry_const<unsigned long long>(a, b, carry_in);
  } else {
    SumCarry<unsigned long long> result{0, 0};
    result.sum = __builtin_addcll(a, b, carry_in, &result.carry);
    return result;
  }
}

#endif // LIBC_HAS_BUILTIN(__builtin_addc)

// Subtract with borrow
template <typename T> struct DiffBorrow {
  T diff;
  T borrow;
};

// This version is always valid for constexpr.
template <typename T>
LIBC_INLINE constexpr cpp::enable_if_t<
    cpp::is_integral_v<T> && cpp::is_unsigned_v<T>, DiffBorrow<T>>
sub_with_borrow_const(T a, T b, T borrow_in) {
  T tmp = a - b;
  T diff = tmp - borrow_in;
  T borrow_out = (diff > tmp) + (tmp > a);
  return {diff, borrow_out};
}

// This version is not always valid for constepxr because it's overriden below
// if builtins are available.
template <typename T>
LIBC_INLINE constexpr cpp::enable_if_t<
    cpp::is_integral_v<T> && cpp::is_unsigned_v<T>, DiffBorrow<T>>
sub_with_borrow(T a, T b, T borrow_in) {
  return sub_with_borrow_const<T>(a, b, borrow_in);
}

#if LIBC_HAS_BUILTIN(__builtin_subc)
// https://clang.llvm.org/docs/LanguageExtensions.html#multiprecision-arithmetic-builtins

template <>
LIBC_INLINE constexpr DiffBorrow<unsigned char>
sub_with_borrow<unsigned char>(unsigned char a, unsigned char b,
                               unsigned char borrow_in) {
  if (__builtin_is_constant_evaluated()) {
    return sub_with_borrow_const<unsigned char>(a, b, borrow_in);
  } else {
    DiffBorrow<unsigned char> result{0, 0};
    result.diff = __builtin_subcb(a, b, borrow_in, &result.borrow);
    return result;
  }
}

template <>
LIBC_INLINE constexpr DiffBorrow<unsigned short>
sub_with_borrow<unsigned short>(unsigned short a, unsigned short b,
                                unsigned short borrow_in) {
  if (__builtin_is_constant_evaluated()) {
    return sub_with_borrow_const<unsigned short>(a, b, borrow_in);
  } else {
    DiffBorrow<unsigned short> result{0, 0};
    result.diff = __builtin_subcs(a, b, borrow_in, &result.borrow);
    return result;
  }
}

template <>
LIBC_INLINE constexpr DiffBorrow<unsigned int>
sub_with_borrow<unsigned int>(unsigned int a, unsigned int b,
                              unsigned int borrow_in) {
  if (__builtin_is_constant_evaluated()) {
    return sub_with_borrow_const<unsigned int>(a, b, borrow_in);
  } else {
    DiffBorrow<unsigned int> result{0, 0};
    result.diff = __builtin_subc(a, b, borrow_in, &result.borrow);
    return result;
  }
}

template <>
LIBC_INLINE constexpr DiffBorrow<unsigned long>
sub_with_borrow<unsigned long>(unsigned long a, unsigned long b,
                               unsigned long borrow_in) {
  if (__builtin_is_constant_evaluated()) {
    return sub_with_borrow_const<unsigned long>(a, b, borrow_in);
  } else {
    DiffBorrow<unsigned long> result{0, 0};
    result.diff = __builtin_subcl(a, b, borrow_in, &result.borrow);
    return result;
  }
}

template <>
LIBC_INLINE constexpr DiffBorrow<unsigned long long>
sub_with_borrow<unsigned long long>(unsigned long long a, unsigned long long b,
                                    unsigned long long borrow_in) {
  if (__builtin_is_constant_evaluated()) {
    return sub_with_borrow_const<unsigned long long>(a, b, borrow_in);
  } else {
    DiffBorrow<unsigned long long> result{0, 0};
    result.diff = __builtin_subcll(a, b, borrow_in, &result.borrow);
    return result;
  }
}

#endif // LIBC_HAS_BUILTIN(__builtin_subc)

} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_SRC___SUPPORT_MATH_EXTRAS_H

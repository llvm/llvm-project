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

#include "src/__support/CPP/bit.h"           // countl_one, countr_zero
#include "src/__support/CPP/limits.h"        // CHAR_BIT, numeric_limits
#include "src/__support/CPP/type_traits.h" // is_unsigned_v, is_constant_evaluated
#include "src/__support/macros/attributes.h" // LIBC_INLINE

namespace LIBC_NAMESPACE {

// Create a bitmask with the count right-most bits set to 1, and all other bits
// set to 0.  Only unsigned types are allowed.
template <typename T, size_t count>
LIBC_INLINE constexpr cpp::enable_if_t<cpp::is_unsigned_v<T>, T>
mask_trailing_ones() {
  constexpr unsigned T_BITS = CHAR_BIT * sizeof(T);
  static_assert(count <= T_BITS && "Invalid bit index");
  return count == 0 ? 0 : (T(-1) >> (T_BITS - count));
}

// Create a bitmask with the count left-most bits set to 1, and all other bits
// set to 0.  Only unsigned types are allowed.
template <typename T, size_t count>
LIBC_INLINE constexpr cpp::enable_if_t<cpp::is_unsigned_v<T>, T>
mask_leading_ones() {
  return T(~mask_trailing_ones<T, CHAR_BIT * sizeof(T) - count>());
}

// Create a bitmask with the count right-most bits set to 0, and all other bits
// set to 1.  Only unsigned types are allowed.
template <typename T, size_t count>
LIBC_INLINE constexpr cpp::enable_if_t<cpp::is_unsigned_v<T>, T>
mask_trailing_zeros() {
  return mask_leading_ones<T, CHAR_BIT * sizeof(T) - count>();
}

// Create a bitmask with the count left-most bits set to 0, and all other bits
// set to 1.  Only unsigned types are allowed.
template <typename T, size_t count>
LIBC_INLINE constexpr cpp::enable_if_t<cpp::is_unsigned_v<T>, T>
mask_leading_zeros() {
  return mask_trailing_ones<T, CHAR_BIT * sizeof(T) - count>();
}

#define RETURN_IF_BUILTIN(TYPE, BUILTIN)                                       \
  if constexpr (cpp::is_same_v<T, TYPE> && LIBC_HAS_BUILTIN(BUILTIN))          \
    return BUILTIN(a, b, res);

template <typename T>
[[nodiscard]] LIBC_INLINE constexpr bool add_overflow(T a, T b, T *res) {
  RETURN_IF_BUILTIN(T, __builtin_add_overflow)
  if (res)
    *res = a + b;
  // https://stackoverflow.com/a/1514309
  return (b > 0 && a > (cpp::numeric_limits<T>::max() - b)) ||
         (b < 0 && a < (cpp::numeric_limits<T>::min() - b));
}

template <typename T>
[[nodiscard]] LIBC_INLINE constexpr bool sub_overflow(T a, T b, T *res) {
  RETURN_IF_BUILTIN(T, __builtin_sub_overflow)
  if (res)
    *res = a - b;
  // https://stackoverflow.com/a/1514309
  return (b < 0 && a > (cpp::numeric_limits<T>::max() + b)) ||
         (b > 0 && a < (cpp::numeric_limits<T>::min() + b));
}

#undef RETURN_IF_BUILTIN
#define RETURN_IF_BUILTIN(TYPE, BUILTIN)                                       \
  if constexpr (cpp::is_same_v<T, TYPE> && LIBC_HAS_BUILTIN(BUILTIN))          \
    return BUILTIN(a, b, carry_in, carry_out);

template <typename T>
[[nodiscard]] LIBC_INLINE constexpr cpp::enable_if_t<cpp::is_unsigned_v<T>, T>
add_with_carry(T a, T b, T carry_in, T *carry_out = nullptr) {
  if constexpr (!cpp::is_constant_evaluated()) {
    RETURN_IF_BUILTIN(unsigned char, __builtin_addcb)
    RETURN_IF_BUILTIN(unsigned short, __builtin_addcs)
    RETURN_IF_BUILTIN(unsigned int, __builtin_addc)
    RETURN_IF_BUILTIN(unsigned long, __builtin_addcl)
    RETURN_IF_BUILTIN(unsigned long long, __builtin_addcll)
  }
  T sum;
  T carry1 = add_overflow(a, b, &sum);
  T carry2 = add_overflow(sum, carry_in, &sum);
  if (carry_out)
    *carry_out = carry1 | carry2;
  return sum;
}

template <typename T>
[[nodiscard]] LIBC_INLINE constexpr cpp::enable_if_t<cpp::is_unsigned_v<T>, T>
sub_with_borrow(T a, T b, T carry_in, T *carry_out = nullptr) {
  if constexpr (!cpp::is_constant_evaluated()) {
    RETURN_IF_BUILTIN(unsigned char, __builtin_subcb)
    RETURN_IF_BUILTIN(unsigned short, __builtin_subcs)
    RETURN_IF_BUILTIN(unsigned int, __builtin_subc)
    RETURN_IF_BUILTIN(unsigned long, __builtin_subcl)
    RETURN_IF_BUILTIN(unsigned long long, __builtin_subcll)
  }
  T sub;
  T carry1 = sub_overflow(a, b, &sub);
  T carry2 = sub_overflow(sub, carry_in, &sub);
  if (carry_out)
    *carry_out = carry1 | carry2;
  return sub;
}

#undef RETURN_IF

template <typename T>
[[nodiscard]] LIBC_INLINE constexpr cpp::enable_if_t<cpp::is_unsigned_v<T>, int>
first_leading_zero(T value) {
  return value == cpp::numeric_limits<T>::max() ? 0
                                                : cpp::countl_one(value) + 1;
}

template <typename T>
[[nodiscard]] LIBC_INLINE constexpr cpp::enable_if_t<cpp::is_unsigned_v<T>, int>
first_leading_one(T value) {
  return first_leading_zero(static_cast<T>(~value));
}

template <typename T>
[[nodiscard]] LIBC_INLINE constexpr cpp::enable_if_t<cpp::is_unsigned_v<T>, int>
first_trailing_zero(T value) {
  return value == cpp::numeric_limits<T>::max()
             ? 0
             : cpp::countr_zero(static_cast<T>(~value)) + 1;
}

template <typename T>
[[nodiscard]] LIBC_INLINE constexpr cpp::enable_if_t<cpp::is_unsigned_v<T>, int>
first_trailing_one(T value) {
  return value == cpp::numeric_limits<T>::max() ? 0
                                                : cpp::countr_zero(value) + 1;
}

template <typename T>
[[nodiscard]] LIBC_INLINE constexpr cpp::enable_if_t<cpp::is_unsigned_v<T>, int>
count_zeros(T value) {
  return cpp::popcount<T>(static_cast<T>(~value));
}

} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_SRC___SUPPORT_MATH_EXTRAS_H

//===-- Memory utils --------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_MEMORY_UTILS_UTILS_H
#define LLVM_LIBC_SRC_MEMORY_UTILS_UTILS_H

#include "src/__support/CPP/bit.h"
#include "src/__support/CPP/cstddef.h"
#include "src/__support/CPP/type_traits.h"
#include "src/__support/endian.h"
#include "src/__support/macros/attributes.h" // LIBC_INLINE
#include "src/__support/macros/config.h"     // LIBC_HAS_BUILTIN
#include "src/__support/macros/properties/architectures.h"

#include <stddef.h> // size_t
#include <stdint.h> // intptr_t / uintptr_t / INT32_MAX / INT32_MIN

namespace __llvm_libc {

// Allows compile time error reporting in `if constexpr` branches.
template <bool flag = false>
LIBC_INLINE void deferred_static_assert(const char *msg) {
  static_assert(flag, "compilation error");
  (void)msg;
}

// Return whether `value` is zero or a power of two.
LIBC_INLINE constexpr bool is_power2_or_zero(size_t value) {
  return (value & (value - 1U)) == 0;
}

// Return whether `value` is a power of two.
LIBC_INLINE constexpr bool is_power2(size_t value) {
  return value && is_power2_or_zero(value);
}

// Compile time version of log2 that handles 0.
LIBC_INLINE constexpr size_t log2s(size_t value) {
  return (value == 0 || value == 1) ? 0 : 1 + log2s(value / 2);
}

// Returns the first power of two preceding value or value if it is already a
// power of two (or 0 when value is 0).
LIBC_INLINE constexpr size_t le_power2(size_t value) {
  return value == 0 ? value : 1ULL << log2s(value);
}

// Returns the first power of two following value or value if it is already a
// power of two (or 0 when value is 0).
LIBC_INLINE constexpr size_t ge_power2(size_t value) {
  return is_power2_or_zero(value) ? value : 1ULL << (log2s(value) + 1);
}

// Returns the number of bytes to substract from ptr to get to the previous
// multiple of alignment. If ptr is already aligned returns 0.
template <size_t alignment>
LIBC_INLINE uintptr_t distance_to_align_down(const void *ptr) {
  static_assert(is_power2(alignment), "alignment must be a power of 2");
  return reinterpret_cast<uintptr_t>(ptr) & (alignment - 1U);
}

// Returns the number of bytes to add to ptr to get to the next multiple of
// alignment. If ptr is already aligned returns 0.
template <size_t alignment>
LIBC_INLINE uintptr_t distance_to_align_up(const void *ptr) {
  static_assert(is_power2(alignment), "alignment must be a power of 2");
  // The logic is not straightforward and involves unsigned modulo arithmetic
  // but the generated code is as fast as it can be.
  return -reinterpret_cast<uintptr_t>(ptr) & (alignment - 1U);
}

// Returns the number of bytes to add to ptr to get to the next multiple of
// alignment. If ptr is already aligned returns alignment.
template <size_t alignment>
LIBC_INLINE uintptr_t distance_to_next_aligned(const void *ptr) {
  return alignment - distance_to_align_down<alignment>(ptr);
}

// Returns the same pointer but notifies the compiler that it is aligned.
template <size_t alignment, typename T>
LIBC_INLINE T *assume_aligned(T *ptr) {
  return reinterpret_cast<T *>(__builtin_assume_aligned(ptr, alignment));
}

// Returns true iff memory regions [p1, p1 + size] and [p2, p2 + size] are
// disjoint.
LIBC_INLINE bool is_disjoint(const void *p1, const void *p2, size_t size) {
  const char *a = static_cast<const char *>(p1);
  const char *b = static_cast<const char *>(p2);
  if (a > b) {
    // Swap a and b, this compiles down to conditionnal move for aarch64, x86
    // and RISCV with zbb extension.
    const char *tmp = a;
    a = b;
    b = tmp;
  }
  return a + size <= b;
}

#if LIBC_HAS_BUILTIN(__builtin_memcpy_inline)
#define LLVM_LIBC_HAS_BUILTIN_MEMCPY_INLINE
#endif

#if LIBC_HAS_BUILTIN(__builtin_memset_inline)
#define LLVM_LIBC_HAS_BUILTIN_MEMSET_INLINE
#endif

// Performs a constant count copy.
template <size_t Size>
LIBC_INLINE void memcpy_inline(void *__restrict dst,
                               const void *__restrict src) {
#ifdef LLVM_LIBC_HAS_BUILTIN_MEMCPY_INLINE
  __builtin_memcpy_inline(dst, src, Size);
#else
// In memory functions `memcpy_inline` is instantiated several times with
// different value of the Size parameter. This doesn't play well with GCC's
// Value Range Analysis that wrongly detects out of bounds accesses. We disable
// the 'array-bounds' warning for the purpose of this function.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Warray-bounds"
  for (size_t i = 0; i < Size; ++i)
    static_cast<char *>(dst)[i] = static_cast<const char *>(src)[i];
#pragma GCC diagnostic pop
#endif
}

using Ptr = cpp::byte *;        // Pointer to raw data.
using CPtr = const cpp::byte *; // Const pointer to raw data.

// This type makes sure that we don't accidentally promote an integral type to
// another one. It is only constructible from the exact T type.
template <typename T> struct StrictIntegralType {
  static_assert(cpp::is_integral_v<T>);

  // Can only be constructed from a T.
  template <typename U, cpp::enable_if_t<cpp::is_same_v<U, T>, bool> = 0>
  StrictIntegralType(U value) : value(value) {}

  // Allows using the type in an if statement.
  explicit operator bool() const { return value; }

  // If type is unsigned (bcmp) we allow bitwise OR operations.
  StrictIntegralType operator|(const StrictIntegralType &Rhs) const {
    static_assert(!cpp::is_signed_v<T>);
    return value | Rhs.value;
  }

  // For interation with the C API we allow explicit conversion back to the
  // `int` type.
  explicit operator int() const {
    // bit_cast makes sure that T and int have the same size.
    return cpp::bit_cast<int>(value);
  }

  // Helper to get the zero value.
  LIBC_INLINE static constexpr StrictIntegralType ZERO() { return {T(0)}; }
  LIBC_INLINE static constexpr StrictIntegralType NONZERO() { return {T(1)}; }

private:
  T value;
};

using MemcmpReturnType = StrictIntegralType<int32_t>;
using BcmpReturnType = StrictIntegralType<uint32_t>;

// This implements the semantic of 'memcmp' returning a negative value when 'a'
// is less than 'b', '0' when 'a' equals 'b' and a positive number otherwise.
LIBC_INLINE MemcmpReturnType cmp_uint32_t(uint32_t a, uint32_t b) {
  // We perform the difference as an int64_t.
  const int64_t diff = static_cast<int64_t>(a) - static_cast<int64_t>(b);
  // For the int64_t to int32_t conversion we want the following properties:
  // - int32_t[31:31] == 1 iff diff < 0
  // - int32_t[31:0] == 0 iff diff == 0

  // We also observe that:
  // - When diff < 0: diff[63:32] == 0xffffffff and diff[31:0] != 0
  // - When diff > 0: diff[63:32] == 0 and diff[31:0] != 0
  // - When diff == 0: diff[63:32] == 0 and diff[31:0] == 0
  // - https://godbolt.org/z/8W7qWP6e5
  // - This implies that we can only look at diff[32:32] for determining the
  // sign bit for the returned int32_t.

  // So, we do the following:
  // - int32_t[31:31] = diff[32:32]
  // - int32_t[30:0] = diff[31:0] == 0 ? 0 : non-0.

  // And, we can achieve the above by the expression below. We could have also
  // used (diff64 >> 1) | (diff64 & 0x1) but (diff64 & 0xFFFF) is faster than
  // (diff64 & 0x1). https://godbolt.org/z/j3b569rW1
  return static_cast<int32_t>((diff >> 1) | (diff & 0xFFFF));
}

// Returns a negative value if 'a' is less than 'b' and a positive value
// otherwise. This implements the semantic of 'memcmp' when we know that 'a' and
// 'b' differ.
LIBC_INLINE MemcmpReturnType cmp_neq_uint64_t(uint64_t a, uint64_t b) {
#if defined(LIBC_TARGET_ARCH_IS_X86_64)
  // On x86, the best strategy would be to use 'INT32_MAX' and 'INT32_MIN' for
  // positive and negative value respectively as they are one value apart:
  //   xor     eax, eax         <- free
  //   cmp     rdi, rsi         <- serializing
  //   adc     eax, 2147483647  <- serializing

  // Unfortunately we found instances of client code that negate the result of
  // 'memcmp' to reverse ordering. Because signed integers are not symmetric
  // (e.g., int8_t ∈ [-128, 127]) returning 'INT_MIN' would break such code as
  // `-INT_MIN` is not representable as an int32_t.

  // As a consequence, we use 5 and -5 which is still OK nice in terms of
  // latency.
  //   cmp     rdi, rsi         <- serializing
  //   mov     ecx, -5          <- can be done in parallel
  //   mov     eax, 5           <- can be done in parallel
  //   cmovb   eax, ecx         <- serializing
  static constexpr int32_t POSITIVE = 5;
  static constexpr int32_t NEGATIVE = -5;
#else
  // On RISC-V we simply use '1' and '-1' as it leads to branchless code.
  // On ARMv8, both strategies lead to the same performance.
  static constexpr int32_t POSITIVE = 1;
  static constexpr int32_t NEGATIVE = -1;
#endif
  static_assert(POSITIVE > 0);
  static_assert(NEGATIVE < 0);
  return a < b ? NEGATIVE : POSITIVE;
}

// Loads bytes from memory (possibly unaligned) and materializes them as
// type.
template <typename T> LIBC_INLINE T load(CPtr ptr) {
  T Out;
  memcpy_inline<sizeof(T)>(&Out, ptr);
  return Out;
}

// Stores a value of type T in memory (possibly unaligned).
template <typename T> LIBC_INLINE void store(Ptr ptr, T value) {
  memcpy_inline<sizeof(T)>(ptr, &value);
}

// On architectures that do not allow for unaligned access we perform several
// aligned accesses and recombine them through shifts and logicals operations.
// For instance, if we know that the pointer is 2-byte aligned we can decompose
// a 64-bit operation into four 16-bit operations.

// Loads a 'ValueType' by decomposing it into several loads that are assumed to
// be aligned.
// e.g. load_aligned<uint32_t, uint16_t, uint16_t>(ptr);
template <typename ValueType, typename T, typename... TS>
ValueType load_aligned(CPtr src) {
  static_assert(sizeof(ValueType) >= (sizeof(T) + ... + sizeof(TS)));
  const ValueType value = load<T>(assume_aligned<sizeof(T)>(src));
  if constexpr (sizeof...(TS) > 0) {
    constexpr size_t shift = sizeof(T) * 8;
    const ValueType next = load_aligned<ValueType, TS...>(src + sizeof(T));
    if constexpr (Endian::IS_LITTLE)
      return value | (next << shift);
    else if constexpr (Endian::IS_BIG)
      return (value << shift) | next;
    else
      deferred_static_assert("Invalid endianness");
  } else {
    return value;
  }
}

// Alias for loading a 'uint32_t'.
template <typename T, typename... TS>
auto load32_aligned(CPtr src, size_t offset) {
  static_assert((sizeof(T) + ... + sizeof(TS)) == sizeof(uint32_t));
  return load_aligned<uint32_t, T, TS...>(src + offset);
}

// Alias for loading a 'uint64_t'.
template <typename T, typename... TS>
auto load64_aligned(CPtr src, size_t offset) {
  static_assert((sizeof(T) + ... + sizeof(TS)) == sizeof(uint64_t));
  return load_aligned<uint64_t, T, TS...>(src + offset);
}

// Stores a 'ValueType' by decomposing it into several stores that are assumed
// to be aligned.
// e.g. store_aligned<uint32_t, uint16_t, uint16_t>(value, ptr);
template <typename ValueType, typename T, typename... TS>
void store_aligned(ValueType value, Ptr dst) {
  static_assert(sizeof(ValueType) >= (sizeof(T) + ... + sizeof(TS)));
  constexpr size_t shift = sizeof(T) * 8;
  if constexpr (Endian::IS_LITTLE) {
    store<T>(assume_aligned<sizeof(T)>(dst), value & ~T(0));
    if constexpr (sizeof...(TS) > 0)
      store_aligned<ValueType, TS...>(value >> shift, dst + sizeof(T));
  } else if constexpr (Endian::IS_BIG) {
    constexpr size_t OFFSET = (0 + ... + sizeof(TS));
    store<T>(assume_aligned<sizeof(T)>(dst + OFFSET), value & ~T(0));
    if constexpr (sizeof...(TS) > 0)
      store_aligned<ValueType, TS...>(value >> shift, dst);
  } else {
    deferred_static_assert("Invalid endianness");
  }
}

// Alias for storing a 'uint32_t'.
template <typename T, typename... TS>
void store32_aligned(uint32_t value, Ptr dst, size_t offset) {
  static_assert((sizeof(T) + ... + sizeof(TS)) == sizeof(uint32_t));
  store_aligned<uint32_t, T, TS...>(value, dst + offset);
}

// Alias for storing a 'uint64_t'.
template <typename T, typename... TS>
void store64_aligned(uint64_t value, Ptr dst, size_t offset) {
  static_assert((sizeof(T) + ... + sizeof(TS)) == sizeof(uint64_t));
  store_aligned<uint64_t, T, TS...>(value, dst + offset);
}

// Advances the pointers p1 and p2 by offset bytes and decrease count by the
// same amount.
template <typename T1, typename T2>
LIBC_INLINE void adjust(ptrdiff_t offset, T1 *__restrict &p1,
                        T2 *__restrict &p2, size_t &count) {
  p1 += offset;
  p2 += offset;
  count -= offset;
}

// Advances p1 and p2 so p1 gets aligned to the next SIZE bytes boundary
// and decrease count by the same amount.
// We make sure the compiler knows about the adjusted pointer alignment.
template <size_t SIZE, typename T1, typename T2>
void align_p1_to_next_boundary(T1 *__restrict &p1, T2 *__restrict &p2,
                               size_t &count) {
  adjust(distance_to_next_aligned<SIZE>(p1), p1, p2, count);
  p1 = assume_aligned<SIZE>(p1);
}

// Same as align_p1_to_next_boundary above but with a single pointer instead.
template <size_t SIZE, typename T1>
void align_to_next_boundary(T1 *&p1, size_t &count) {
  CPtr dummy;
  align_p1_to_next_boundary<SIZE>(p1, dummy, count);
}

// An enum class that discriminates between the first and second pointer.
enum class Arg { P1, P2, Dst = P1, Src = P2 };

// Same as align_p1_to_next_boundary but allows for aligning p2 instead of p1.
// Precondition: &p1 != &p2
template <size_t SIZE, Arg AlignOn, typename T1, typename T2>
void align_to_next_boundary(T1 *__restrict &p1, T2 *__restrict &p2,
                            size_t &count) {
  if constexpr (AlignOn == Arg::P1)
    align_p1_to_next_boundary<SIZE>(p1, p2, count);
  else if constexpr (AlignOn == Arg::P2)
    align_p1_to_next_boundary<SIZE>(p2, p1, count); // swapping p1 and p2.
  else
    deferred_static_assert("AlignOn must be either Arg::P1 or Arg::P2");
}

template <size_t SIZE> struct AlignHelper {
  AlignHelper(CPtr ptr) : offset_(distance_to_next_aligned<SIZE>(ptr)) {}

  LIBC_INLINE bool not_aligned() const { return offset_ != SIZE; }
  LIBC_INLINE uintptr_t offset() const { return offset_; }

private:
  uintptr_t offset_;
};

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_MEMORY_UTILS_UTILS_H

//===-- Memory Size ---------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/bit.h" // has_single_bit
#include "src/__support/CPP/limits.h"
#include "src/__support/CPP/type_traits.h"
#include "src/__support/macros/attributes.h"
#include "src/__support/macros/optimization.h"
#include "src/string/memory_utils/utils.h"

namespace LIBC_NAMESPACE {
namespace internal {
template <class T> LIBC_INLINE bool mul_overflow(T a, T b, T *res) {
#if LIBC_HAS_BUILTIN(__builtin_mul_overflow)
  return __builtin_mul_overflow(a, b, res);
#else
  T max = cpp::numeric_limits<T>::max();
  T min = cpp::numeric_limits<T>::min();
  bool overflow = (b > 0 && (a > max / b || a < min / b)) ||
                  (b < 0 && (a < max / b || a > min / b));
  if (!overflow)
    *res = a * b;
  return overflow;
#endif
}
// Limit memory size to the max of ssize_t
class SafeMemSize {
private:
  using type = cpp::make_signed_t<size_t>;
  type value;
  LIBC_INLINE explicit SafeMemSize(type value) : value(value) {}

public:
  LIBC_INLINE_VAR static constexpr size_t MAX_MEM_SIZE =
      static_cast<size_t>(cpp::numeric_limits<type>::max());

  LIBC_INLINE explicit SafeMemSize(size_t value)
      : value(value <= MAX_MEM_SIZE ? static_cast<type>(value) : -1) {}

  LIBC_INLINE static constexpr size_t offset_to(size_t val, size_t align) {
    return (-val) & (align - 1);
  }

  LIBC_INLINE operator size_t() { return static_cast<size_t>(value); }

  LIBC_INLINE bool valid() { return value >= 0; }

  LIBC_INLINE SafeMemSize operator+(const SafeMemSize &other) {
    type result;
    if (LIBC_UNLIKELY((value | other.value) < 0))
      result = -1;
    result = value + other.value;
    return SafeMemSize{result};
  }

  LIBC_INLINE SafeMemSize operator*(const SafeMemSize &other) {
    type result;
    if (LIBC_UNLIKELY((value | other.value) < 0))
      result = -1;
    if (LIBC_UNLIKELY(mul_overflow(value, other.value, &result)))
      result = -1;
    return SafeMemSize{result};
  }

  LIBC_INLINE SafeMemSize align_up(size_t alignment) {
    if (!cpp::has_single_bit(alignment) || alignment > MAX_MEM_SIZE || !valid())
      return SafeMemSize{type{-1}};

    type offset = offset_to(value, alignment);

    if (LIBC_UNLIKELY(offset > static_cast<type>(MAX_MEM_SIZE) - value))
      return SafeMemSize{type{-1}};

    return SafeMemSize{value + offset};
  }
};
} // namespace internal
} // namespace LIBC_NAMESPACE

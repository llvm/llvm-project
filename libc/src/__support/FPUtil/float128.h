//===-- Definition for Float128 data type -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_FPUTIL_FLOAT128_H
#define LLVM_LIBC_SRC___SUPPORT_FPUTIL_FLOAT128_H

#include "src/__support/CPP/type_traits.h"
#include "src/__support/FPUtil/cast.h"
#include "src/__support/FPUtil/comparison_operations.h"
#include "src/__support/macros/attributes.h"
#include "src/__support/macros/config.h"
#include "src/__support/uint128.h"
#include <stdint.h>

namespace LIBC_NAMESPACE_DECL {
namespace fputil {

struct Float128 {
  UInt128 bits;

  LIBC_INLINE Float128() = default;

  template <typename T> LIBC_INLINE constexpr explicit Float128(T x) : bits(0) {
    if constexpr (cpp::is_floating_point_v<T>) {
      bits = fputil::cast<Float128>(x).bits;
    }
    // TODO: add rem after testing
  }

  template <typename T, cpp::enable_if_t<cpp::is_floating_point_v<T> &&
                                             !cpp::is_same_v<T, Float128>,
                                         int> = 0>
  LIBC_INLINE LIBC_CONSTEXPR_DEFAULT operator T() const {
    return fputil::cast<T>(*this);
  }
  LIBC_INLINE constexpr bool operator==(Float128 &other) const {
    return fputil::equals(*this, other);
  }

  LIBC_INLINE constexpr bool operator!=(Float128 &other) const {
    return !fputil::equals(*this, other);
  }

  LIBC_INLINE constexpr bool operator<(Float128 &other) const {
    return fputil::less_than(*this, other);
  }

  LIBC_INLINE constexpr bool operator<=(Float128 &other) const {
    return fputil::less_than_or_equals(*this, other);
  }

  LIBC_INLINE constexpr bool operator>(Float128 &other) const {
    return fputil::greater_than(*this, other);
  }

  LIBC_INLINE constexpr bool operator>=(Float128 &other) const {
    return fputil::greater_than_or_equals(*this, other);
  }
};

} // namespace fputil
} // namespace LIBC_NAMESPACE_DECL

static_assert(
    LIBC_NAMESPACE::cpp::is_trivially_constructible<
        LIBC_NAMESPACE::fputil::Float128>::value);
static_assert(
    LIBC_NAMESPACE::cpp::is_trivially_copyable<
        LIBC_NAMESPACE::fputil::Float128>::value);

#endif // LLVM_LIBC_SRC___SUPPORT_FPUTIL_FLOAT128_H

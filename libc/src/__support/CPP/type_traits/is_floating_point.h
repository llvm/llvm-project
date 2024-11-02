//===-- is_floating_point type_traits ---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_LIBC_SRC___SUPPORT_CPP_TYPE_TRAITS_IS_FLOATING_POINT_H
#define LLVM_LIBC_SRC___SUPPORT_CPP_TYPE_TRAITS_IS_FLOATING_POINT_H

#include "src/__support/CPP/type_traits/is_same.h"
#include "src/__support/CPP/type_traits/remove_cv.h"
#include "src/__support/macros/attributes.h"
#include "src/__support/macros/properties/types.h" // LIBC_TYPES_HAS_FLOAT128

namespace LIBC_NAMESPACE::cpp {

// is_floating_point
template <typename T> struct is_floating_point {
private:
  template <typename Head, typename... Args>
  LIBC_INLINE_VAR static constexpr bool __is_unqualified_any_of() {
    return (... || is_same_v<remove_cv_t<Head>, Args>);
  }

public:
#if defined(LIBC_TYPES_HAS_FLOAT128)
  LIBC_INLINE_VAR static constexpr bool value =
      __is_unqualified_any_of<T, float, double, long double, float128>();
#else
  LIBC_INLINE_VAR static constexpr bool value =
      __is_unqualified_any_of<T, float, double, long double>();
#endif // LIBC_TYPES_HAS_FLOAT128
};
template <typename T>
LIBC_INLINE_VAR constexpr bool is_floating_point_v =
    is_floating_point<T>::value;

} // namespace LIBC_NAMESPACE::cpp

#endif // LLVM_LIBC_SRC___SUPPORT_CPP_TYPE_TRAITS_IS_FLOATING_POINT_H

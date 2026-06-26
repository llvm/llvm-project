//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TEST_SF_CMATH_COMMON_H
#define TEST_SF_CMATH_COMMON_H

#include <cassert>
#include <cerrno>
#include <cmath>
#include <math.h>
#include <cfenv>

// std::type_identity is C++20 (we need to support C++17 here)
template <class T>
struct type_identity {
  typedef T type;
};
template <class T>
using type_identity_t = typename type_identity<T>::type;

template <class T>
bool between(type_identity_t<T> lower, T value, type_identity_t<T> upper) {
  return lower < value && value < upper;
}

template <class Func>
void check_no_domain_error(Func f) {
#if math_errhandling & MATH_ERRNO
  errno = EACCES;
#endif
#if math_errhandling & MATH_ERREXCEPT
  std::feclearexcept(FE_INVALID);
#endif
  f();
#if math_errhandling & MATH_ERRNO
  assert(errno == EACCES);
#endif
#if math_errhandling & MATH_ERREXCEPT
  assert(!std::fetestexcept(FE_INVALID));
#endif
}

template <class Func>
void check_domain_error(Func f) {
#if math_errhandling & MATH_ERRNO
  errno = EACCES;
#endif
#if math_errhandling & MATH_ERREXCEPT
  std::feclearexcept(FE_INVALID);
#endif
  f();
#if math_errhandling & MATH_ERRNO
  assert(errno == EDOM);
#endif
#if math_errhandling & MATH_ERREXCEPT
  assert(std::fetestexcept(FE_INVALID));
#endif
}

#endif // TEST_SF_CMATH_COMMON_H

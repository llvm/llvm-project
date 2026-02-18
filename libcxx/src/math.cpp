//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifdef __APPLE__
#  define _REENTRANT
#endif

#include <cmath>
#include <math.h> // for lgamma_r

_LIBCPP_BEGIN_NAMESPACE_STD

__lgamma_result __lgamma_thread_safe_impl(double __d) noexcept {
  int __sign;
  double __result = ::lgamma_r(__d, &__sign);
  return __lgamma_result{__result, __sign};
}

_LIBCPP_END_NAMESPACE_STD

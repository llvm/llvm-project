//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Picolibc does not define some of the floating point environment macros for
// arm platforms without hardware floating point support.
// UNSUPPORTED: LIBCXX-PICOLIBC-FIXME

// <fenv.h>

#include <fenv.h>

#include "test_macros.h"

#ifndef FE_DIVBYZERO
#error FE_DIVBYZERO not defined
#endif

#ifndef FE_INEXACT
#error FE_INEXACT not defined
#endif

#ifndef FE_INVALID
#error FE_INVALID not defined
#endif

#ifndef FE_OVERFLOW
#error FE_OVERFLOW not defined
#endif

#ifndef FE_UNDERFLOW
#error FE_UNDERFLOW not defined
#endif

#ifndef FE_ALL_EXCEPT
#error FE_ALL_EXCEPT not defined
#endif

#ifndef FE_DOWNWARD
#error FE_DOWNWARD not defined
#endif

#ifndef FE_TONEAREST
#error FE_TONEAREST not defined
#endif

#ifndef FE_TOWARDZERO
#error FE_TOWARDZERO not defined
#endif

#ifndef FE_UPWARD
#error FE_UPWARD not defined
#endif

#ifndef FE_DFL_ENV
#error FE_DFL_ENV not defined
#endif

fenv_t fenv = {};
fexcept_t fex = fexcept_t();
ASSERT_SAME_TYPE(int, decltype(::feclearexcept(0)));
ASSERT_SAME_TYPE(int, decltype(::fegetexceptflag(&fex, 0)));
ASSERT_SAME_TYPE(int, decltype(::feraiseexcept(0)));
ASSERT_SAME_TYPE(int, decltype(::fesetexceptflag(&fex, 0)));
ASSERT_SAME_TYPE(int, decltype(::fetestexcept(0)));
ASSERT_SAME_TYPE(int, decltype(::fegetround()));
ASSERT_SAME_TYPE(int, decltype(::fesetround(0)));
ASSERT_SAME_TYPE(int, decltype(::fegetenv(&fenv)));
ASSERT_SAME_TYPE(int, decltype(::feholdexcept(&fenv)));
ASSERT_SAME_TYPE(int, decltype(::fesetenv(&fenv)));
ASSERT_SAME_TYPE(int, decltype(::feupdateenv(&fenv)));

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <fenv.h>

#include <fenv.h>
#include <type_traits>

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

int main(int, char**)
{
    fenv_t fenv = {};
    fexcept_t fex = 0;
    static_assert((std::is_same<decltype(::feclearexcept(0)), int>::value), "");
    static_assert((std::is_same<decltype(::fegetexceptflag(&fex, 0)), int>::value), "");
    static_assert((std::is_same<decltype(::feraiseexcept(0)), int>::value), "");
    static_assert((std::is_same<decltype(::fesetexceptflag(&fex, 0)), int>::value), "");
    static_assert((std::is_same<decltype(::fetestexcept(0)), int>::value), "");
    static_assert((std::is_same<decltype(::fegetround()), int>::value), "");
    static_assert((std::is_same<decltype(::fesetround(0)), int>::value), "");
    static_assert((std::is_same<decltype(::fegetenv(&fenv)), int>::value), "");
    static_assert((std::is_same<decltype(::feholdexcept(&fenv)), int>::value), "");
    static_assert((std::is_same<decltype(::fesetenv(&fenv)), int>::value), "");
    static_assert((std::is_same<decltype(::feupdateenv(&fenv)), int>::value), "");

  return 0;
}

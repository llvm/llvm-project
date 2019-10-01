//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: newlib

// <cfenv>

#include <cfenv>
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
    std::fenv_t fenv;
    std::fexcept_t fex;
    ((void)fenv); // Prevent unused warning
    ((void)fex); // Prevent unused warning
    static_assert((std::is_same<decltype(std::feclearexcept(0)), int>::value), "");
    static_assert((std::is_same<decltype(std::fegetexceptflag(&fex, 0)), int>::value), "");
    static_assert((std::is_same<decltype(std::feraiseexcept(0)), int>::value), "");
    static_assert((std::is_same<decltype(std::fesetexceptflag(&fex, 0)), int>::value), "");
    static_assert((std::is_same<decltype(std::fetestexcept(0)), int>::value), "");
    static_assert((std::is_same<decltype(std::fegetround()), int>::value), "");
    static_assert((std::is_same<decltype(std::fesetround(0)), int>::value), "");
    static_assert((std::is_same<decltype(std::fegetenv(&fenv)), int>::value), "");
    static_assert((std::is_same<decltype(std::feholdexcept(&fenv)), int>::value), "");
    static_assert((std::is_same<decltype(std::fesetenv(&fenv)), int>::value), "");
    static_assert((std::is_same<decltype(std::feupdateenv(&fenv)), int>::value), "");

  return 0;
}

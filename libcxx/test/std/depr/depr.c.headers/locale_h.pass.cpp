//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <locale.h>

#include <locale.h>
#include <type_traits>

#include "test_macros.h"

#ifndef LC_ALL
#error LC_ALL not defined
#endif

#ifndef LC_COLLATE
#error LC_COLLATE not defined
#endif

#ifndef LC_CTYPE
#error LC_CTYPE not defined
#endif

#ifndef LC_MONETARY
#error LC_MONETARY not defined
#endif

#ifndef LC_NUMERIC
#error LC_NUMERIC not defined
#endif

#ifndef LC_TIME
#error LC_TIME not defined
#endif

#ifndef NULL
#error NULL not defined
#endif

int main(int, char**)
{
    lconv lc; ((void)lc);
    static_assert((std::is_same<decltype(setlocale(0, "")), char*>::value), "");
    static_assert((std::is_same<decltype(localeconv()), lconv*>::value), "");

  return 0;
}

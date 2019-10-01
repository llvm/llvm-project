//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <ctype.h>

#include <ctype.h>
#include <type_traits>
#include <cassert>

#include "test_macros.h"

#ifdef isalnum
#error isalnum defined
#endif

#ifdef isalpha
#error isalpha defined
#endif

#ifdef isblank
#error isblank defined
#endif

#ifdef iscntrl
#error iscntrl defined
#endif

#ifdef isdigit
#error isdigit defined
#endif

#ifdef isgraph
#error isgraph defined
#endif

#ifdef islower
#error islower defined
#endif

#ifdef isprint
#error isprint defined
#endif

#ifdef ispunct
#error ispunct defined
#endif

#ifdef isspace
#error isspace defined
#endif

#ifdef isupper
#error isupper defined
#endif

#ifdef isxdigit
#error isxdigit defined
#endif

#ifdef tolower
#error tolower defined
#endif

#ifdef toupper
#error toupper defined
#endif

int main(int, char**)
{
    static_assert((std::is_same<decltype(isalnum(0)), int>::value), "");
    static_assert((std::is_same<decltype(isalpha(0)), int>::value), "");
    static_assert((std::is_same<decltype(isblank(0)), int>::value), "");
    static_assert((std::is_same<decltype(iscntrl(0)), int>::value), "");
    static_assert((std::is_same<decltype(isdigit(0)), int>::value), "");
    static_assert((std::is_same<decltype(isgraph(0)), int>::value), "");
    static_assert((std::is_same<decltype(islower(0)), int>::value), "");
    static_assert((std::is_same<decltype(isprint(0)), int>::value), "");
    static_assert((std::is_same<decltype(ispunct(0)), int>::value), "");
    static_assert((std::is_same<decltype(isspace(0)), int>::value), "");
    static_assert((std::is_same<decltype(isupper(0)), int>::value), "");
    static_assert((std::is_same<decltype(isxdigit(0)), int>::value), "");
    static_assert((std::is_same<decltype(tolower(0)), int>::value), "");
    static_assert((std::is_same<decltype(toupper(0)), int>::value), "");

    assert(isalnum('a'));
    assert(isalpha('a'));
    assert(isblank(' '));
    assert(!iscntrl(' '));
    assert(!isdigit('a'));
    assert(isgraph('a'));
    assert(islower('a'));
    assert(isprint('a'));
    assert(!ispunct('a'));
    assert(!isspace('a'));
    assert(!isupper('a'));
    assert(isxdigit('a'));
    assert(tolower('A') == 'a');
    assert(toupper('a') == 'A');

  return 0;
}

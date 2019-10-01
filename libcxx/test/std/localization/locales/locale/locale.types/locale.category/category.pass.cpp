//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This test uses new symbols that were not defined in the libc++ shipped on
// darwin11 and darwin12:
// XFAIL: availability=macosx10.7
// XFAIL: availability=macosx10.8

// <locale>

// typedef int category;

#include <locale>
#include <type_traits>
#include <cassert>

#include "test_macros.h"

template <class T>
void test(const T &) {}


int main(int, char**)
{
    static_assert((std::is_same<std::locale::category, int>::value), "");
    assert(std::locale::none == 0);
    assert(std::locale::collate);
    assert(std::locale::ctype);
    assert(std::locale::monetary);
    assert(std::locale::numeric);
    assert(std::locale::time);
    assert(std::locale::messages);
    assert((std::locale::collate
          & std::locale::ctype
          & std::locale::monetary
          & std::locale::numeric
          & std::locale::time
          & std::locale::messages) == 0);
    assert((std::locale::collate
          | std::locale::ctype
          | std::locale::monetary
          | std::locale::numeric
          | std::locale::time
          | std::locale::messages)
         == std::locale::all);

    test(std::locale::none);
    test(std::locale::collate);
    test(std::locale::ctype);
    test(std::locale::monetary);
    test(std::locale::numeric);
    test(std::locale::time);
    test(std::locale::messages);
    test(std::locale::all);

  return 0;
}

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <locale>

// template <> class codecvt<char32_t, char, mbstate_t>

// int length(stateT& state, const externT* from, const externT* from_end, size_t max) const;

// This test runs in C++20, but we have deprecated codecvt<char(16|32), char, mbstate_t> in C++20.
// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DISABLE_DEPRECATION_WARNINGS

#include <locale>
#include <cassert>

#include "test_macros.h"

typedef std::codecvt<char32_t, char, std::mbstate_t> F;

int main(int, char**)
{
    std::locale l = std::locale::classic();
    const F& f = std::use_facet<F>(l);
    std::mbstate_t mbs = {};
    const char from[] = "some text";
    assert(f.length(mbs, from, from+10, 0) == 0);
    assert(f.length(mbs, from, from+10, 8) == 8);
    assert(f.length(mbs, from, from+10, 9) == 9);
    assert(f.length(mbs, from, from+10, 10) == 10);
    assert(f.length(mbs, from, from+10, 100) == 10);

  return 0;
}

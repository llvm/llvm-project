//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <locale>

// template <> class codecvt<char16_t, char, mbstate_t>

// result in(stateT& state,
//           const externT* from, const externT* from_end, const externT*& from_next,
//           internT* to, internT* to_end, internT*& to_next) const;

// This test runs in C++20, but we have deprecated codecvt<char(16|32), char, mbstate_t> in C++20.
// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DISABLE_DEPRECATION_WARNINGS

#include <locale>
#include <string>
#include <vector>
#include <cassert>

#include "test_macros.h"

typedef std::codecvt<char16_t, char, std::mbstate_t> F;

int main(int, char**)
{
    std::locale l = std::locale::classic();
    const char from[] = "some text";
    F::intern_type to[9];
    const F& f = std::use_facet<F>(l);
    std::mbstate_t mbs = {};
    const char* from_next = 0;
    F::intern_type* to_next = 0;
    assert(f.in(mbs, from, from + 9, from_next,
                     to, to + 9, to_next) == F::ok);
    assert(from_next - from == 9);
    assert(to_next - to == 9);
    for (unsigned i = 0; i < 9; ++i)
        assert(to[i] == from[i]);

  return 0;
}

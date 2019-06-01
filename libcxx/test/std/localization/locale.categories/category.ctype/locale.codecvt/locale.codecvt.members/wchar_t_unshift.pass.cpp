//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <locale>

// template <> class codecvt<wchar_t, char, mbstate_t>

// result unshift(stateT& state,
//                externT* to, externT* to_end, externT*& to_next) const;

// This is pretty much just an "are you breathing" test

#include <locale>
#include <string>
#include <vector>
#include <cassert>

#include "test_macros.h"

typedef std::codecvt<wchar_t, char, std::mbstate_t> F;

int main(int, char**)
{
    std::locale l = std::locale::classic();
    std::vector<F::extern_type> to(3);
    const F& f = std::use_facet<F>(l);
    std::mbstate_t mbs = {};
    F::extern_type* to_next = 0;
    assert(f.unshift(mbs, to.data(), to.data() + to.size(), to_next) == F::ok);
    assert(to_next == to.data());

  return 0;
}

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <locale>

// template <> class codecvt<char16_t, char, mbstate_t>

// bool always_noconv() const throw();

#include <locale>
#include <cassert>

#include "test_macros.h"

typedef std::codecvt<char16_t, char, std::mbstate_t> F;

int main(int, char**)
{
    std::locale l = std::locale::classic();
    const F& f = std::use_facet<F>(l);
    assert(!f.always_noconv());

  return 0;
}

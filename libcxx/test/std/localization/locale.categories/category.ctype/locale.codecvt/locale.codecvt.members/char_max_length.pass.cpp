//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <locale>

// template <> class codecvt<char, char, mbstate_t>

// int max_length() const throw();

#include <locale>
#include <cassert>

typedef std::codecvt<char, char, std::mbstate_t> F;

int main(int, char**)
{
    std::locale l = std::locale::classic();
    const F& f = std::use_facet<F>(l);
    assert(f.max_length() == 1);

  return 0;
}

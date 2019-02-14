//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <locale>

// template <> class codecvt<char, char, mbstate_t>

// result out(stateT& state,
//            const internT* from, const internT* from_end, const internT*& from_next,
//            externT* to, externT* to_end, externT*& to_next) const;

#include <locale>
#include <string>
#include <vector>
#include <cassert>

typedef std::codecvt<char, char, std::mbstate_t> F;

int main(int, char**)
{
    std::locale l = std::locale::classic();
    const std::basic_string<F::intern_type> from("some text");
    std::vector<char> to(from.size());
    const F& f = std::use_facet<F>(l);
    std::mbstate_t mbs = {};
    const char* from_next = 0;
    char* to_next = 0;
    assert(f.out(mbs, from.data(), from.data() + from.size(), from_next,
                      to.data(), to.data() + to.size(), to_next) == F::noconv);
    assert(from_next == from.data());
    assert(to_next == to.data());

  return 0;
}

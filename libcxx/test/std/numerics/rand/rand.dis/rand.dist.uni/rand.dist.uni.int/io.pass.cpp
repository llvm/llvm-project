//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <random>

// template<class _IntType = int>
// class uniform_int_distribution

// template <class charT, class traits>
// basic_ostream<charT, traits>&
// operator<<(basic_ostream<charT, traits>& os,
//            const uniform_int_distribution& x);
//
// template <class charT, class traits>
// basic_istream<charT, traits>&
// operator>>(basic_istream<charT, traits>& is,
//            uniform_int_distribution& x);

#include <random>
#include <sstream>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    {
        typedef std::uniform_int_distribution<> D;
        D d1(3, 8);
        std::ostringstream os;
        os << d1;
        std::istringstream is(os.str());
        D d2;
        is >> d2;
        assert(d1 == d2);
    }

  return 0;
}

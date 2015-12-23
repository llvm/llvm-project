//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <utility>

// template <class T1, class T2> struct pair

// constexpr pair();

// NOTE: The SFINAE on the default constructor is tested in
//       default-sfinae.pass.cpp

#include <utility>
#include <type_traits>
#include <cassert>

#include "test_macros.h"

int main()
{
    {
    typedef std::pair<float, short*> P;
    P p;
    assert(p.first == 0.0f);
    assert(p.second == nullptr);
    }
#if TEST_STD_VER >= 11
    {
        typedef std::pair<float, short*> P;
        constexpr P p;
        static_assert(p.first == 0.0f, "");
        static_assert(p.second == nullptr, "");
    }
#endif
}

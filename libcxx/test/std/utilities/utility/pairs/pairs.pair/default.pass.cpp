//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <utility>

// template <class T1, class T2> struct pair

// constexpr pair();

// This test doesn't pass due to a constexpr bug in GCC 4.9 that fails
// to initialize any type without a user provided constructor in a constant
// expression (e.g. float).
// XFAIL: gcc-4.9

// NOTE: The SFINAE on the default constructor is tested in
//       default-sfinae.pass.cpp


#include <utility>
#include <type_traits>
#include <cassert>

#include "test_macros.h"
#include "archetypes.hpp"

int main(int, char**)
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
    {
        using NoDefault = ImplicitTypes::NoDefault;
        using P = std::pair<int, NoDefault>;
        static_assert(!std::is_default_constructible<P>::value, "");
        using P2 = std::pair<NoDefault, int>;
        static_assert(!std::is_default_constructible<P2>::value, "");
    }
#endif

  return 0;
}

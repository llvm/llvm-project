//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// __half_positive divides an integer number by 2 as unsigned number for known types.
// It can be an important optimization for lower bound, for example.

#include <algorithm>
#include <cassert>
#include <limits>
#include <type_traits>

#include "test_macros.h"
#include "user_defined_integral.h"

namespace {

template <class IntType, class UnderlyingType = IntType>
TEST_CONSTEXPR bool test(IntType max_v = IntType(std::numeric_limits<UnderlyingType>::max())) {
    return std::__half_positive(max_v) == max_v / 2;
}

}  // namespace

int main(int, char**)
{
    {
        assert(test<char>());
        assert(test<int>());
        assert(test<long>());
        assert((test<UserDefinedIntegral<int>, int>()));
        assert(test<size_t>());
#if !defined(_LIBCPP_HAS_NO_INT128)
        assert(test<__int128_t>());
#endif  // !defined(_LIBCPP_HAS_NO_INT128)
    }

#if TEST_STD_VER >= 11
    {
        static_assert(test<char>(), "");
        static_assert(test<int>(), "");
        static_assert(test<long>(), "");
        static_assert(test<size_t>(), "");
#if !defined(_LIBCPP_HAS_NO_INT128)
        static_assert(test<__int128_t>(), "");
#endif  // !defined(_LIBCPP_HAS_NO_INT128)
    }
#endif // TEST_STD_VER >= 11

  return 0;
}

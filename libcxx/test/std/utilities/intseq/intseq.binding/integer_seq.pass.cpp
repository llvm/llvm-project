//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26
// <utility>

// structured binding support for integer_sequence

#include <tuple>
#include <utility>
#include <type_traits>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    using empty  = std::integer_sequence<int>;
    using size4  = std::integer_sequence<int, 9, 8, 7, 2>;

    static_assert ( std::tuple_size_v<empty> == 0, "empty size wrong" );
    static_assert ( std::tuple_size_v<const empty> == 0, "empty size wrong" );

    static_assert ( std::tuple_size_v<size4> == 4, "size4 size wrong" );
    static_assert ( std::tuple_size_v<const size4> == 4, "size4 size wrong" );

    static_assert ( std::is_same_v<std::tuple_element_t<0, size4>, int>, "size4 type wrong" );
    static_assert ( std::is_same_v<std::tuple_element_t<1, size4>, int>, "size4 type wrong" );
    static_assert ( std::is_same_v<std::tuple_element_t<2, size4>, int>, "size4 type wrong" );
    static_assert ( std::is_same_v<std::tuple_element_t<3, size4>, int>, "size4 type wrong" );

    static_assert ( std::is_same_v<std::tuple_element_t<0, const size4>, int>, "const4 type wrong" );
    static_assert ( std::is_same_v<std::tuple_element_t<1, const size4>, int>, "const4 type wrong" );
    static_assert ( std::is_same_v<std::tuple_element_t<2, const size4>, int>, "const4 type wrong" );
    static_assert ( std::is_same_v<std::tuple_element_t<3, const size4>, int>, "const4 type wrong" );

    constexpr static size4 seq4{};
    static_assert ( get<0> (seq4) == 9, "size4 element 0 wrong" );
    static_assert ( get<1> (seq4) == 8, "size4 element 1 wrong" );
    static_assert ( get<2> (seq4) == 7, "size4 element 2 wrong" );
    static_assert ( get<3> (seq4) == 2, "size4 element 3 wrong" );

  return 0;
}

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11, c++14
// type_traits

// template<class B> struct negation;                        // C++17
// template<class B>
//   constexpr bool negation_v = negation<B>::value;         // C++17

#include <type_traits>
#include <cassert>

#include "test_macros.h"

struct True  { static constexpr bool value = true; };
struct False { static constexpr bool value = false; };

int main(int, char**)
{
    static_assert (!std::negation<std::true_type >::value, "" );
    static_assert ( std::negation<std::false_type>::value, "" );

    static_assert (!std::negation_v<std::true_type >, "" );
    static_assert ( std::negation_v<std::false_type>, "" );

    static_assert (!std::negation<True >::value, "" );
    static_assert ( std::negation<False>::value, "" );

    static_assert (!std::negation_v<True >, "" );
    static_assert ( std::negation_v<False>, "" );

    static_assert ( std::negation<std::negation<std::true_type >>::value, "" );
    static_assert (!std::negation<std::negation<std::false_type>>::value, "" );

  return 0;
}

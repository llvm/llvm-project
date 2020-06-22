//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// type_traits

// template<class... B> struct conjunction;                           // C++17
// template<class... B>
//   constexpr bool conjunction_v = conjunction<B...>::value;         // C++17

#include <type_traits>
#include <cassert>

#include "test_macros.h"

struct True  { static constexpr bool value = true; };
struct False { static constexpr bool value = false; };

int main(int, char**)
{
    static_assert ( std::conjunction<>::value, "" );
    static_assert ( std::conjunction<std::true_type >::value, "" );
    static_assert (!std::conjunction<std::false_type>::value, "" );

    static_assert ( std::conjunction_v<>, "" );
    static_assert ( std::conjunction_v<std::true_type >, "" );
    static_assert (!std::conjunction_v<std::false_type>, "" );

    static_assert ( std::conjunction<std::true_type,  std::true_type >::value, "" );
    static_assert (!std::conjunction<std::true_type,  std::false_type>::value, "" );
    static_assert (!std::conjunction<std::false_type, std::true_type >::value, "" );
    static_assert (!std::conjunction<std::false_type, std::false_type>::value, "" );

    static_assert ( std::conjunction_v<std::true_type,  std::true_type >, "" );
    static_assert (!std::conjunction_v<std::true_type,  std::false_type>, "" );
    static_assert (!std::conjunction_v<std::false_type, std::true_type >, "" );
    static_assert (!std::conjunction_v<std::false_type, std::false_type>, "" );

    static_assert ( std::conjunction<std::true_type,  std::true_type,  std::true_type >::value, "" );
    static_assert (!std::conjunction<std::true_type,  std::false_type, std::true_type >::value, "" );
    static_assert (!std::conjunction<std::false_type, std::true_type,  std::true_type >::value, "" );
    static_assert (!std::conjunction<std::false_type, std::false_type, std::true_type >::value, "" );
    static_assert (!std::conjunction<std::true_type,  std::true_type,  std::false_type>::value, "" );
    static_assert (!std::conjunction<std::true_type,  std::false_type, std::false_type>::value, "" );
    static_assert (!std::conjunction<std::false_type, std::true_type,  std::false_type>::value, "" );
    static_assert (!std::conjunction<std::false_type, std::false_type, std::false_type>::value, "" );

    static_assert ( std::conjunction_v<std::true_type,  std::true_type,  std::true_type >, "" );
    static_assert (!std::conjunction_v<std::true_type,  std::false_type, std::true_type >, "" );
    static_assert (!std::conjunction_v<std::false_type, std::true_type,  std::true_type >, "" );
    static_assert (!std::conjunction_v<std::false_type, std::false_type, std::true_type >, "" );
    static_assert (!std::conjunction_v<std::true_type,  std::true_type,  std::false_type>, "" );
    static_assert (!std::conjunction_v<std::true_type,  std::false_type, std::false_type>, "" );
    static_assert (!std::conjunction_v<std::false_type, std::true_type,  std::false_type>, "" );
    static_assert (!std::conjunction_v<std::false_type, std::false_type, std::false_type>, "" );

    static_assert ( std::conjunction<True >::value, "" );
    static_assert (!std::conjunction<False>::value, "" );

    static_assert ( std::conjunction_v<True >, "" );
    static_assert (!std::conjunction_v<False>, "" );

  return 0;
}

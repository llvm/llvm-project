//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11, c++14
// type_traits

// template<class... B> struct disjunction;                           // C++17
// template<class... B>
//   constexpr bool disjunction_v = disjunction<B...>::value;         // C++17

#include <type_traits>
#include <cassert>

struct True  { static constexpr bool value = true; };
struct False { static constexpr bool value = false; };

int main()
{
    static_assert (!std::disjunction<>::value, "" );
    static_assert ( std::disjunction<std::true_type >::value, "" );
    static_assert (!std::disjunction<std::false_type>::value, "" );

    static_assert (!std::disjunction_v<>, "" );
    static_assert ( std::disjunction_v<std::true_type >, "" );
    static_assert (!std::disjunction_v<std::false_type>, "" );

    static_assert ( std::disjunction<std::true_type,  std::true_type >::value, "" );
    static_assert ( std::disjunction<std::true_type,  std::false_type>::value, "" );
    static_assert ( std::disjunction<std::false_type, std::true_type >::value, "" );
    static_assert (!std::disjunction<std::false_type, std::false_type>::value, "" );

    static_assert ( std::disjunction_v<std::true_type,  std::true_type >, "" );
    static_assert ( std::disjunction_v<std::true_type,  std::false_type>, "" );
    static_assert ( std::disjunction_v<std::false_type, std::true_type >, "" );
    static_assert (!std::disjunction_v<std::false_type, std::false_type>, "" );

    static_assert ( std::disjunction<std::true_type,  std::true_type,  std::true_type >::value, "" );
    static_assert ( std::disjunction<std::true_type,  std::false_type, std::true_type >::value, "" );
    static_assert ( std::disjunction<std::false_type, std::true_type,  std::true_type >::value, "" );
    static_assert ( std::disjunction<std::false_type, std::false_type, std::true_type >::value, "" );
    static_assert ( std::disjunction<std::true_type,  std::true_type,  std::false_type>::value, "" );
    static_assert ( std::disjunction<std::true_type,  std::false_type, std::false_type>::value, "" );
    static_assert ( std::disjunction<std::false_type, std::true_type,  std::false_type>::value, "" );
    static_assert (!std::disjunction<std::false_type, std::false_type, std::false_type>::value, "" );

    static_assert ( std::disjunction_v<std::true_type,  std::true_type,  std::true_type >, "" );
    static_assert ( std::disjunction_v<std::true_type,  std::false_type, std::true_type >, "" );
    static_assert ( std::disjunction_v<std::false_type, std::true_type,  std::true_type >, "" );
    static_assert ( std::disjunction_v<std::false_type, std::false_type, std::true_type >, "" );
    static_assert ( std::disjunction_v<std::true_type,  std::true_type,  std::false_type>, "" );
    static_assert ( std::disjunction_v<std::true_type,  std::false_type, std::false_type>, "" );
    static_assert ( std::disjunction_v<std::false_type, std::true_type,  std::false_type>, "" );
    static_assert (!std::disjunction_v<std::false_type, std::false_type, std::false_type>, "" );

    static_assert ( std::disjunction<True >::value, "" );
    static_assert (!std::disjunction<False>::value, "" );

    static_assert ( std::disjunction_v<True >, "" );
    static_assert (!std::disjunction_v<False>, "" );
}

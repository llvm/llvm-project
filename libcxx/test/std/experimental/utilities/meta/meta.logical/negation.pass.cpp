//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11
// <experimental/type_traits>

// template<class B> struct negation;
// template<class B>
//   constexpr bool negation_v = negation<B>::value;

#include <experimental/type_traits>
#include <cassert>

namespace ex = std::experimental;

struct True  { static constexpr bool value = true; };
struct False { static constexpr bool value = false; };

int main()
{
    static_assert (!ex::negation<std::true_type >::value, "" );
    static_assert ( ex::negation<std::false_type>::value, "" );

    static_assert (!ex::negation_v<std::true_type >, "" );
    static_assert ( ex::negation_v<std::false_type>, "" );

    static_assert (!ex::negation<True >::value, "" );
    static_assert ( ex::negation<False>::value, "" );

    static_assert (!ex::negation_v<True >, "" );
    static_assert ( ex::negation_v<False>, "" );

    static_assert ( ex::negation<ex::negation<std::true_type >>::value, "" );
    static_assert (!ex::negation<ex::negation<std::false_type>>::value, "" );
}

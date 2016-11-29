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

// template<class... B> struct disjunction;
// template<class... B>
//   constexpr bool disjunction_v = disjunction<B...>::value;

#include <experimental/type_traits>
#include <cassert>

namespace ex = std::experimental;

struct True  { static constexpr bool value = true; };
struct False { static constexpr bool value = false; };

int main()
{
    static_assert (!ex::disjunction<>::value, "" );
    static_assert ( ex::disjunction<std::true_type >::value, "" );
    static_assert (!ex::disjunction<std::false_type>::value, "" );

    static_assert (!ex::disjunction_v<>, "" );
    static_assert ( ex::disjunction_v<std::true_type >, "" );
    static_assert (!ex::disjunction_v<std::false_type>, "" );

    static_assert ( ex::disjunction<std::true_type,  std::true_type >::value, "" );
    static_assert ( ex::disjunction<std::true_type,  std::false_type>::value, "" );
    static_assert ( ex::disjunction<std::false_type, std::true_type >::value, "" );
    static_assert (!ex::disjunction<std::false_type, std::false_type>::value, "" );

    static_assert ( ex::disjunction_v<std::true_type,  std::true_type >, "" );
    static_assert ( ex::disjunction_v<std::true_type,  std::false_type>, "" );
    static_assert ( ex::disjunction_v<std::false_type, std::true_type >, "" );
    static_assert (!ex::disjunction_v<std::false_type, std::false_type>, "" );

    static_assert ( ex::disjunction<std::true_type,  std::true_type,  std::true_type >::value, "" );
    static_assert ( ex::disjunction<std::true_type,  std::false_type, std::true_type >::value, "" );
    static_assert ( ex::disjunction<std::false_type, std::true_type,  std::true_type >::value, "" );
    static_assert ( ex::disjunction<std::false_type, std::false_type, std::true_type >::value, "" );
    static_assert ( ex::disjunction<std::true_type,  std::true_type,  std::false_type>::value, "" );
    static_assert ( ex::disjunction<std::true_type,  std::false_type, std::false_type>::value, "" );
    static_assert ( ex::disjunction<std::false_type, std::true_type,  std::false_type>::value, "" );
    static_assert (!ex::disjunction<std::false_type, std::false_type, std::false_type>::value, "" );

    static_assert ( ex::disjunction_v<std::true_type,  std::true_type,  std::true_type >, "" );
    static_assert ( ex::disjunction_v<std::true_type,  std::false_type, std::true_type >, "" );
    static_assert ( ex::disjunction_v<std::false_type, std::true_type,  std::true_type >, "" );
    static_assert ( ex::disjunction_v<std::false_type, std::false_type, std::true_type >, "" );
    static_assert ( ex::disjunction_v<std::true_type,  std::true_type,  std::false_type>, "" );
    static_assert ( ex::disjunction_v<std::true_type,  std::false_type, std::false_type>, "" );
    static_assert ( ex::disjunction_v<std::false_type, std::true_type,  std::false_type>, "" );
    static_assert (!ex::disjunction_v<std::false_type, std::false_type, std::false_type>, "" );

    static_assert ( ex::disjunction<True >::value, "" );
    static_assert (!ex::disjunction<False>::value, "" );

    static_assert ( ex::disjunction_v<True >, "" );
    static_assert (!ex::disjunction_v<False>, "" );
}

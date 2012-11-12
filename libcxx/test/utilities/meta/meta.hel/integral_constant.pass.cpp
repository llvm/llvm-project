//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// type_traits

// integral_constant

#include <type_traits>
#include <cassert>

int main()
{
    typedef std::integral_constant<int, 5> _5;
    static_assert(_5::value == 5, "");
    static_assert((std::is_same<_5::value_type, int>::value), "");
    static_assert((std::is_same<_5::type, _5>::value), "");
#ifndef _LIBCPP_HAS_NO_CONSTEXPR
    static_assert((_5() == 5), "");
#else
    assert(_5() == 5);
#endif

    static_assert(std::false_type::value == false, "");
    static_assert((std::is_same<std::false_type::value_type, bool>::value), "");
    static_assert((std::is_same<std::false_type::type, std::false_type>::value), "");

    static_assert(std::true_type::value == true, "");
    static_assert((std::is_same<std::true_type::value_type, bool>::value), "");
    static_assert((std::is_same<std::true_type::type, std::true_type>::value), "");

    std::false_type f1;
    std::false_type f2 = f1;

    std::true_type t1;
    std::true_type t2 = t1;
}

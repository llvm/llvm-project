//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <utility>

// template <class T> typename add_rvalue_reference<T>::type declval() noexcept;

#include <utility>
#include <type_traits>

class A
{
    A(const A&);
    A& operator=(const A&);
};

int main()
{
#ifndef _LIBCPP_HAS_NO_RVALUE_REFERENCES
    static_assert((std::is_same<decltype(std::declval<A>()), A&&>::value), "");
#else
    static_assert((std::is_same<decltype(std::declval<A>()), A>::value), "");
#endif
}

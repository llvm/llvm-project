//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <functional>

// template<class T> struct is_bind_expression

#include <functional>

template <bool Expected, class T>
void
test(const T&)
{
    static_assert(std::is_bind_expression<T>::value == Expected, "");
}

struct C {};

int main()
{
    test<true>(std::bind(C()));
    test<true>(std::bind(C(), std::placeholders::_2));
    test<true>(std::bind<int>(C()));
    test<false>(1);
    test<false>(std::placeholders::_2);
}

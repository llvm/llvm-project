//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <memory>

// template <class T>
// struct pointer_traits<T*>
// {
//     template <class U> using rebind = U*;
//     ...
// };

#include <memory>
#include <type_traits>

int main()
{
#ifndef _LIBCPP_HAS_NO_TEMPLATE_ALIASES
    static_assert((std::is_same<std::pointer_traits<int*>::rebind<double>, double*>::value), "");
#else
    static_assert((std::is_same<std::pointer_traits<int*>::rebind<double>::other, double*>::value), "");
#endif
}

//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <memory>

// template <class Ptr>
// struct pointer_traits
// {
//     template <class U> using rebind = <details>;
//     ...
// };

#include <memory>
#include <type_traits>

template <class T>
struct A
{
};

template <class T> struct B1 {};

template <class T>
struct B
{
#ifndef _LIBCPP_HAS_NO_TEMPLATE_ALIASES
    template <class U> using rebind = B1<U>;
#else
    template <class U> struct rebind {typedef B1<U> other;};
#endif
};

template <class T, class U>
struct C
{
};

template <class T, class U> struct D1 {};

template <class T, class U>
struct D
{
#ifndef _LIBCPP_HAS_NO_TEMPLATE_ALIASES
    template <class V> using rebind = D1<V, U>;
#else
    template <class V> struct rebind {typedef D1<V, U> other;};
#endif
};

int main()
{
#ifndef _LIBCPP_HAS_NO_TEMPLATE_ALIASES
    static_assert((std::is_same<std::pointer_traits<A<int*> >::rebind<double*>, A<double*> >::value), "");
    static_assert((std::is_same<std::pointer_traits<B<int> >::rebind<double>, B1<double> >::value), "");
    static_assert((std::is_same<std::pointer_traits<C<char, int> >::rebind<double>, C<double, int> >::value), "");
    static_assert((std::is_same<std::pointer_traits<D<char, int> >::rebind<double>, D1<double, int> >::value), "");
#else  // _LIBCPP_HAS_NO_TEMPLATE_ALIASES
    static_assert((std::is_same<std::pointer_traits<A<int*> >::rebind<double*>::other, A<double*> >::value), "");
    static_assert((std::is_same<std::pointer_traits<B<int> >::rebind<double>::other, B1<double> >::value), "");
    static_assert((std::is_same<std::pointer_traits<C<char, int> >::rebind<double>::other, C<double, int> >::value), "");
    static_assert((std::is_same<std::pointer_traits<D<char, int> >::rebind<double>::other, D1<double, int> >::value), "");
#endif  // _LIBCPP_HAS_NO_TEMPLATE_ALIASES
}

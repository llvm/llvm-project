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
//     typedef <details> element_type;
//     ...
// };

#include <memory>
#include <type_traits>

struct A
{
    typedef char element_type;
};

template <class T>
struct B
{
    typedef char element_type;
};

template <class T>
struct C
{
};

template <class T, class U>
struct D
{
};

int main()
{
    static_assert((std::is_same<std::pointer_traits<A>::element_type, char>::value), "");
    static_assert((std::is_same<std::pointer_traits<B<int> >::element_type, char>::value), "");
    static_assert((std::is_same<std::pointer_traits<C<int> >::element_type, int>::value), "");
    static_assert((std::is_same<std::pointer_traits<D<double, int> >::element_type, double>::value), "");
}

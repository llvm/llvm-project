//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>

// template <class Ptr>
// struct pointer_traits
// {
//     static pointer pointer_to(<details>);
//     ...
// };

#include <memory>
#include <cassert>

template <class T>
struct A
{
private:
    struct nat {};
public:
    typedef T element_type;
    element_type* t_;

    A(element_type* t) : t_(t) {}

    static A pointer_to(typename std::conditional<std::is_void<element_type>::value,
                                           nat, element_type>::type& et)
        {return A(&et);}
};

int main()
{
    {
        int i = 0;
        static_assert((std::is_same<A<int>, decltype(std::pointer_traits<A<int> >::pointer_to(i))>::value), "");
        A<int> a = std::pointer_traits<A<int> >::pointer_to(i);
        assert(a.t_ == &i);
    }
    {
        (std::pointer_traits<A<void> >::element_type)0;
    }
}

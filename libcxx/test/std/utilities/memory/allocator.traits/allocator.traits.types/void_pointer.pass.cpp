//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>

// template <class Alloc>
// struct allocator_traits
// {
//     typedef Alloc::void_pointer
//           | pointer_traits<pointer>::rebind<void>
//                                          void_pointer;
//     ...
// };

#include <memory>
#include <type_traits>
#include "test_macros.h"

template <class T>
struct Ptr {};

template <class T>
struct A
{
    typedef T value_type;
    typedef Ptr<T> pointer;
};

template <class T>
struct B
{
    typedef T value_type;
};

template <class T>
struct CPtr {};

template <class T>
struct C
{
    typedef T value_type;
    typedef CPtr<void> void_pointer;
};


template <class T>
struct D
{
    typedef T value_type;
private:
    typedef void void_pointer;
};

int main(int, char**)
{
    static_assert((std::is_same<std::allocator_traits<A<char> >::void_pointer, Ptr<void> >::value), "");
    static_assert((std::is_same<std::allocator_traits<B<char> >::void_pointer, void*>::value), "");
    static_assert((std::is_same<std::allocator_traits<C<char> >::void_pointer, CPtr<void> >::value), "");
#if TEST_STD_VER >= 11
    static_assert((std::is_same<std::allocator_traits<D<char> >::void_pointer, void*>::value), "");
#endif

  return 0;
}

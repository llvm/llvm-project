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
//     typedef Alloc::const_pointer
//           | pointer_traits<pointer>::rebind<const value_type>
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
    typedef CPtr<T> pointer;
    typedef CPtr<const T> const_pointer;
};

template <class T>
struct D {
  typedef T value_type;
private:
  typedef void const_pointer;
};

int main()
{
    static_assert((std::is_same<std::allocator_traits<A<char> >::const_pointer, Ptr<const char> >::value), "");
    static_assert((std::is_same<std::allocator_traits<B<char> >::const_pointer, const char*>::value), "");
    static_assert((std::is_same<std::allocator_traits<C<char> >::const_pointer, CPtr<const char> >::value), "");
#if TEST_STD_VER >= 11
    static_assert((std::is_same<std::allocator_traits<D<char> >::const_pointer, const char*>::value), "");
#endif
}

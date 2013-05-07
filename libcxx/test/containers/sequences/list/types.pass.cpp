//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <list>

// template <class T, class Alloc = allocator<T> >
// class list
// {
// public:
//
//     // types:
//     typedef T value_type;
//     typedef Alloc allocator_type;
//     typedef typename allocator_type::reference reference;
//     typedef typename allocator_type::const_reference const_reference;
//     typedef typename allocator_type::pointer pointer;
//     typedef typename allocator_type::const_pointer const_pointer;

#include <list>
#include <type_traits>

int main()
{
    static_assert((std::is_same<std::list<int>::value_type, int>::value), "");
    static_assert((std::is_same<std::list<int>::allocator_type, std::allocator<int> >::value), "");
    static_assert((std::is_same<std::list<int>::reference, std::allocator<int>::reference>::value), "");
    static_assert((std::is_same<std::list<int>::const_reference, std::allocator<int>::const_reference>::value), "");
    static_assert((std::is_same<std::list<int>::pointer, std::allocator<int>::pointer>::value), "");
    static_assert((std::is_same<std::list<int>::const_pointer, std::allocator<int>::const_pointer>::value), "");
}

//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <memory>

// template <class Alloc>
// struct allocator_traits
// {
//     static size_type max_size(const allocator_type& a);
//     ...
// };

#include <memory>
#include <new>
#include <type_traits>
#include <cassert>

template <class T>
struct A
{
    typedef T value_type;

};

template <class T>
struct B
{
    typedef T value_type;

    size_t max_size() const
    {
        return 100;
    }
};

int main()
{
#ifndef _LIBCPP_HAS_NO_ADVANCED_SFINAE
    {
        A<int> a;
        assert(std::allocator_traits<A<int> >::max_size(a) ==
               std::numeric_limits<std::size_t>::max());
    }
    {
        const A<int> a = {};
        assert(std::allocator_traits<A<int> >::max_size(a) ==
               std::numeric_limits<std::size_t>::max());
    }
#endif  // _LIBCPP_HAS_NO_ADVANCED_SFINAE
    {
        B<int> b;
        assert(std::allocator_traits<B<int> >::max_size(b) == 100);
    }
    {
        const B<int> b = {};
        assert(std::allocator_traits<B<int> >::max_size(b) == 100);
    }
}

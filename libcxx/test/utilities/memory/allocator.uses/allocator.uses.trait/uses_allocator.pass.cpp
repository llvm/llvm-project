//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <memory>

// template <class T, class Alloc> struct uses_allocator;

#include <memory>
#include <vector>

struct A
{
};

struct B
{
    typedef int allocator_type;
};

int main()
{
    static_assert((!std::uses_allocator<int, std::allocator<int> >::value), "");
    static_assert(( std::uses_allocator<std::vector<int>, std::allocator<int> >::value), "");
    static_assert((!std::uses_allocator<A, std::allocator<int> >::value), "");
    static_assert((!std::uses_allocator<B, std::allocator<int> >::value), "");
    static_assert(( std::uses_allocator<B, double>::value), "");
}

//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11
// dynarray.data

// template <class Type, class Alloc>
//   struct uses_allocator<dynarray<Type>, Alloc> : true_type { };


#include <__config>

#include <experimental/dynarray>
#include "test_allocator.h"

using std::experimental::dynarray;

int main()
{
    static_assert ( std::uses_allocator<dynarray<int>, test_allocator<int>>::value, "" );
}


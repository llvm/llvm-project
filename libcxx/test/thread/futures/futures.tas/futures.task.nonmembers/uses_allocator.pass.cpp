//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <future>

// class packaged_task<R(ArgTypes...)>

// template <class Callable, class Alloc>
//   struct uses_allocator<packaged_task<Callable>, Alloc>
//      : true_type { };

#include <future>
#include "../../test_allocator.h"

int main()
{
    static_assert((std::uses_allocator<std::packaged_task<double(int, char)>, test_allocator<int> >::value), "");
}

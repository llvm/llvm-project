//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: libcpp-has-no-threads

// <future>

// class promise<R>

// template <class R, class Alloc>
//   struct uses_allocator<promise<R>, Alloc>
//      : true_type { };

#include <future>
#include "test_macros.h"
#include "test_allocator.h"

int main(int, char**)
{
    static_assert((std::uses_allocator<std::promise<int>, test_allocator<int> >::value), "");
    static_assert((std::uses_allocator<std::promise<int&>, test_allocator<int> >::value), "");
    static_assert((std::uses_allocator<std::promise<void>, test_allocator<void> >::value), "");

  return 0;
}

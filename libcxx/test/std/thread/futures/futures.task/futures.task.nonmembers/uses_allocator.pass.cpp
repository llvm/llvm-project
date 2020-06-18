//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: libcpp-has-no-threads

// This test is marked XFAIL and not UNSUPPORTED because the non-variadic
// declaration of packaged_task is available in C++03. Therefore the test
// should fail because the static_assert fires and not because std::packaged_task
// in undefined.
// XFAIL: c++03

// <future>
// REQUIRES: c++11 || c++14
// packaged_task allocator support was removed in C++17 (LWG 2976)

// class packaged_task<R(ArgTypes...)>

// template <class Callable, class Alloc>
//   struct uses_allocator<packaged_task<Callable>, Alloc>
//      : true_type { };

#include <future>
#include "test_macros.h"
#include "test_allocator.h"

int main(int, char**)
{
    static_assert((std::uses_allocator<std::packaged_task<double(int, char)>, test_allocator<int> >::value), "");

  return 0;
}

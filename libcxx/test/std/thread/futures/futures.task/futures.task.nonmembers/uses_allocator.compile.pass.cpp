//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: libcpp-has-no-threads

// packaged_task allocator support was removed in C++17 (LWG 2976)
// REQUIRES: c++11 || c++14

// <future>

// class packaged_task<R(ArgTypes...)>

// template <class Callable, class Alloc>
//   struct uses_allocator<packaged_task<Callable>, Alloc>
//      : true_type { };

#include <future>
#include "test_macros.h"
#include "test_allocator.h"

static_assert((std::uses_allocator<std::packaged_task<double(int, char)>, test_allocator<int> >::value), "");

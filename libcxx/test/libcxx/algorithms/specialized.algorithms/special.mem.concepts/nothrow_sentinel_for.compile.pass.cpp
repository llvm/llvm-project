//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts

// template<class S, class I>
// concept __nothrow_sentinel_for;

#include <memory>

static_assert(std::ranges::__nothrow_sentinel_for<int*, int*>);
static_assert(!std::ranges::__nothrow_sentinel_for<int*, long*>);

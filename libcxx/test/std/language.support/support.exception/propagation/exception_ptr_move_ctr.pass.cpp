//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: no-exceptions, c++03

// <exception>

// typedef unspecified exception_ptr;

// Test the move constructor of exception_ptr

#include <exception>
#include <utility>
#include <cassert>

#include "test_macros.h"

int main(int, char**) {
  std::exception_ptr p = std::make_exception_ptr(42);
  std::exception_ptr p2{p};
  assert(p2 == p);
  // Under test: The move constructor
  std::exception_ptr p3{std::move(p2)};
  assert(p3 == p);
// `p2` was moved from. In libc++ it will be nullptr, but
// this is not guaranteed by the standard.
#if defined(_LIBCPP_VERSION) && !defined(_LIBCPP_ABI_MICROSOFT)
  assert(p2 == nullptr);
#endif

  try {
    std::rethrow_exception(p3);
  } catch (int e) {
    assert(e == 42);
  }

  return 0;
}

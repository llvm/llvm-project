//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: no-exceptions

// <exception>

// typedef unspecified exception_ptr;

// Test swapping of exception_ptr

#include <exception>
#include <utility>
#include <cassert>

#include "test_macros.h"

int main(int, char**) {
  std::exception_ptr p21 = std::make_exception_ptr(42);
  std::exception_ptr p42 = std::make_exception_ptr(21);
  std::swap(p42, p21);

  try {
    std::rethrow_exception(p21);
  } catch (int e) {
    assert(e == 21);
  }
  try {
    std::rethrow_exception(p42);
  } catch (int e) {
    assert(e == 42);
  }

  return 0;
}

//===----------------------------------------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// const char* what() const noexcept override;

#include <expected>
#include <cassert>
#include <utility>

#include "test_macros.h"

struct Foo {};

int main(int, char**) {
  {
    std::bad_expected_access<int> const exc(99);
    char const* what = exc.what();
    assert(what != nullptr);
    ASSERT_NOEXCEPT(exc.what());
  }
  {
    std::bad_expected_access<Foo> const exc(Foo{});
    char const* what = exc.what();
    assert(what != nullptr);
    ASSERT_NOEXCEPT(exc.what());
  }

  return 0;
}

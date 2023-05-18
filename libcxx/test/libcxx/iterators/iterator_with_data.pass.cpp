//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

#include "test_macros.h"

TEST_DIAGNOSTIC_PUSH
TEST_CLANG_DIAGNOSTIC_IGNORED("-Wprivate-header")
#include <__iterator/iterator_with_data.h>
TEST_DIAGNOSTIC_POP

#include "test_iterators.h"

static_assert(std::forward_iterator<std::__iterator_with_data<forward_iterator<int*>, int>>);
static_assert(std::bidirectional_iterator<std::__iterator_with_data<bidirectional_iterator<int*>, int>>);
static_assert(std::bidirectional_iterator<std::__iterator_with_data<random_access_iterator<int*>, int>>);
static_assert(std::bidirectional_iterator<std::__iterator_with_data<contiguous_iterator<int*>, int>>);

constexpr bool test() {
  {
    std::__iterator_with_data<forward_iterator<int*>, int> iter(forward_iterator<int*>(nullptr), 3);
    assert(iter == iter);
    assert(iter.__get_iter() == forward_iterator<int*>(nullptr));
    assert(std::move(iter).__get_data() == 3);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}

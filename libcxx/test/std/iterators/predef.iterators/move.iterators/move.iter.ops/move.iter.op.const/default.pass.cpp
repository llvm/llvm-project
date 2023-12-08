//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <iterator>

// move_iterator

// move_iterator();
//
//  constexpr in C++17
//
//  requires the underlying iterator to be default-constructible (extension).

#include <iterator>

#include <type_traits>
#include "test_macros.h"
#include "test_iterators.h"

#if TEST_STD_VER > 17
struct NoDefaultCtr : forward_iterator<int*> {
  NoDefaultCtr() = delete;
};

LIBCPP_STATIC_ASSERT( std::is_default_constructible_v<std::move_iterator<forward_iterator<int*>>>);
LIBCPP_STATIC_ASSERT(!std::is_default_constructible_v<std::move_iterator<NoDefaultCtr>>);
#endif

template <class It>
void test() {
    std::move_iterator<It> r;
    (void)r;
}

int main(int, char**) {
  // we don't have a test iterator that is both input and default-constructible, so not testing that case
  test<forward_iterator<char*> >();
  test<bidirectional_iterator<char*> >();
  test<random_access_iterator<char*> >();
  test<char*>();

#if TEST_STD_VER > 14
  {
    constexpr std::move_iterator<const char *> it;
    (void)it;
  }
#endif

  return 0;
}

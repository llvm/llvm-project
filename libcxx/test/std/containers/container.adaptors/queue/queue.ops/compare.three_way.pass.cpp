//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: GCC-ALWAYS_INLINE-FIXME

// <queue>

// template<class T, three_way_comparable Container>
//   compare_three_way_result_t<Container>
//     operator<=>(const queue<T, Container>& x, const queue<T, Container>& y);

#include <cassert>
#include <deque>
#include <queue>
#include <list>

#include "nasty_containers.h"
#include "test_container_comparisons.h"

int main(int, char**) {
  assert((test_sequence_container_adaptor_spaceship<std::queue, std::deque>()));
  assert((test_sequence_container_adaptor_spaceship<std::queue, std::list>()));
  assert((test_sequence_container_adaptor_spaceship<std::queue, nasty_list>()));
  // `std::queue` is not constexpr, so no `static_assert` test here.
  return 0;
}

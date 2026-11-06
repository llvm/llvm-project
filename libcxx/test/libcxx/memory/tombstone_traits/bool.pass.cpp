//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++20

// tombstone_traits<bool>

#include <__memory/tombstone_traits.h>
#include <cassert>
#include <utility>

struct Empty {};

constexpr bool test() {
  std::__tombstoned_value<bool, Empty> tombstone(std::__init_disengaged);

  assert(!tombstone.__is_engaged());

  tombstone.__engage(std::in_place, true);

  assert(tombstone.__get_value());
  assert(std::as_const(tombstone.__get_value()));
  assert(std::move(tombstone.__get_value()));
  assert(std::move(std::as_const(tombstone.__get_value())));

  tombstone.__disengage(std::in_place);
  assert(!tombstone.__is_engaged());

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}

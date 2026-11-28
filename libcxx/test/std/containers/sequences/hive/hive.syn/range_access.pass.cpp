//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26
// UNSUPPORTED: libcpp-has-no-experimental-hive

// <hive>

// Including <hive> makes the [iterator.range] function templates available (LWG4234). Only <hive> is included.

#include <cassert>
#include <hive>

#include "test_macros.h"

int main(int, char**) {
  std::hive<int> h;
  const std::hive<int>& ch = h;
  assert(std::begin(h) == std::end(h));
  assert(std::begin(ch) == std::end(ch));
  assert(std::cbegin(h) == std::cend(h));
  assert(std::rbegin(h) == std::rend(h));
  assert(std::rbegin(ch) == std::rend(ch));
  assert(std::crbegin(h) == std::crend(h));
  assert(std::size(h) == 0);
  assert(std::ssize(h) == 0);
  assert(std::empty(h));

  h.insert(1);
  h.insert(2);
  assert(std::begin(h) == h.begin());
  assert(std::end(h) == h.end());
  assert(std::cbegin(h) == h.cbegin());
  assert(std::cend(h) == h.cend());
  assert(std::rbegin(h) == h.rbegin());
  assert(std::rend(h) == h.rend());
  assert(std::crbegin(h) == h.crbegin());
  assert(std::crend(h) == h.crend());
  assert(std::size(h) == 2);
  assert(std::ssize(h) == 2);
  assert(!std::empty(h));

  return 0;
}

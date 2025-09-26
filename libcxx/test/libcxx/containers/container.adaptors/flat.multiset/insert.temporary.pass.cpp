//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <flat_set>

// https://https://llvm.org/PR119016

#include <flat_set>

#include <cassert>
#include <utility>
#include <vector>

#include "../flat_helpers.h"
#include "test_macros.h"

bool test() {
  using M = std::flat_multiset<TrackCopyMove>;
  {
    M m;
    TrackCopyMove t;
    m.insert(t);
    assert(m.begin()->copy_count == 1);
    assert(m.begin()->move_count == 0);
  }
  {
    M m;
    TrackCopyMove t;
    m.emplace(t);
    assert(m.begin()->copy_count == 1);
    assert(m.begin()->move_count == 0);
  }

  return true;
}

int main(int, char**) {
  test();

  return 0;
}

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Ensure that all the elements in the vector are destroyed in reverse order on destruction.

// UNSUPPORTED: c++03

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <vector>

#include "test_macros.h"

struct DestroyTracker {
  TEST_CONSTEXPR_CXX20 DestroyTracker(size_t& tracker) : index_(tracker++), tracker_(tracker) {}

  TEST_CONSTEXPR_CXX20 DestroyTracker(const DestroyTracker& other)
      : index_(other.tracker_++), tracker_(other.tracker_) {}

  TEST_CONSTEXPR_CXX20 DestroyTracker& operator=(const DestroyTracker&) { return *this; }
  TEST_CONSTEXPR_CXX20 ~DestroyTracker() {
    --tracker_;
    assert(tracker_ == index_);
  }

  size_t index_;
  size_t& tracker_;
};

TEST_CONSTEXPR_CXX20 bool test() {
  size_t tracker = 0;
  {
    std::vector<DestroyTracker> vec;
    vec.reserve(10);
    for (size_t i = 0; i != 10; ++i)
      vec.emplace_back(tracker);
  }
  assert(tracker == 0);

  return true;
}

int main(int, char**) {
  test();
#if TEST_STD_VER >= 20
  static_assert(test());
#endif

  return 0;
}

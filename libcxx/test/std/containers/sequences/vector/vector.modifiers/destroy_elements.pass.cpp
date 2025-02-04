//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Ensure that all the elements in the vector are destroyed, especially when reallocating the internal buffer

// UNSUPPORTED: c++03

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <vector>

#include "test_macros.h"

struct DestroyTracker {
  TEST_CONSTEXPR_CXX20 DestroyTracker(std::vector<bool>& vec) : vec_(&vec), index_(vec.size()) { vec.push_back(false); }

  TEST_CONSTEXPR_CXX20 DestroyTracker(const DestroyTracker& other) : vec_(other.vec_), index_(vec_->size()) {
    vec_->push_back(false);
  }

  TEST_CONSTEXPR_CXX20 DestroyTracker& operator=(const DestroyTracker&) { return *this; }
  TEST_CONSTEXPR_CXX20 ~DestroyTracker() { (*vec_)[index_] = true; }

  std::vector<bool>* vec_;
  size_t index_;
};

template <class Operation>
TEST_CONSTEXPR_CXX20 void test(Operation operation) {
  std::vector<bool> all_destroyed;

  {
    std::vector<DestroyTracker> v;
    for (size_t i = 0; i != 100; ++i)
      operation(v, all_destroyed);
  }

  assert(std::all_of(all_destroyed.begin(), all_destroyed.end(), [](bool b) { return b; }));
}

TEST_CONSTEXPR_CXX20 bool test() {
  test([](std::vector<DestroyTracker>& vec, std::vector<bool>& tracker) { vec.emplace_back(tracker); });
  test([](std::vector<DestroyTracker>& vec, std::vector<bool>& tracker) { vec.push_back(tracker); });
  test([](std::vector<DestroyTracker>& vec, std::vector<bool>& tracker) { vec.emplace(vec.begin(), tracker); });
  test([](std::vector<DestroyTracker>& vec, std::vector<bool>& tracker) { vec.insert(vec.begin(), tracker); });
  test([](std::vector<DestroyTracker>& vec, std::vector<bool>& tracker) { vec.resize(vec.size() + 1, tracker); });
#if TEST_STD_VER >= 23
  test([](std::vector<DestroyTracker>& vec, std::vector<bool>& tracker) {
    vec.insert_range(vec.begin(), std::array<DestroyTracker, 2>{tracker, tracker});
  });

  test([](std::vector<DestroyTracker>& vec, std::vector<bool>& tracker) {
    vec.append_range(std::array<DestroyTracker, 2>{tracker, tracker});
  });
#endif

  return true;
}

int main() {
  test();
#if TEST_STD_VER >= 20
  static_assert(test());
#endif
}

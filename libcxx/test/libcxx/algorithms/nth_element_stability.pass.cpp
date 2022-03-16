//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// Test std::nth_element stability randomization

// UNSUPPORTED: libcxx-no-debug-mode, c++03, windows
// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DEBUG=1

#include <algorithm>
#include <array>
#include <cassert>
#include <functional>
#include <iterator>
#include <vector>

#include "test_macros.h"

struct MyType {
  int value = 0;
  constexpr bool operator<(const MyType& other) const { return value < other.value; }
};

std::vector<MyType> deterministic() {
  static constexpr int kSize = 100;
  std::vector<MyType> v;
  v.resize(kSize);
  for (int i = 0; i < kSize; ++i) {
    v[i].value = (i % 2 ? i : kSize / 2 + i);
  }
  std::__nth_element(v.begin(), v.begin() + kSize / 2, v.end(), std::less<MyType>());
  return v;
}

void test_randomization() {
  static constexpr int kSize = 100;
  std::vector<MyType> v;
  v.resize(kSize);
  for (int i = 0; i < kSize; ++i) {
    v[i].value = (i % 2 ? i : kSize / 2 + i);
  }
  auto deterministic_v = deterministic();
  std::nth_element(v.begin(), v.begin() + kSize / 2, v.end());
  bool all_equal = true;
  for (int i = 0; i < kSize; ++i) {
    if (v[i].value != deterministic_v[i].value) {
      all_equal = false;
    }
  }
  assert(!all_equal);
}

void test_same() {
  static constexpr int kSize = 100;
  std::vector<MyType> v;
  v.resize(kSize);
  for (int i = 0; i < kSize; ++i) {
    v[i].value = (i % 2 ? i : kSize / 2 + i);
  }
  auto snapshot_v = v;
  auto snapshot_custom_v = v;
  std::nth_element(v.begin(), v.begin() + kSize / 2, v.end());
  std::nth_element(snapshot_v.begin(), snapshot_v.begin() + kSize / 2, snapshot_v.end());
  std::nth_element(snapshot_custom_v.begin(), snapshot_custom_v.begin() + kSize / 2, snapshot_custom_v.end(), std::less<MyType>());
  bool all_equal = true;
  for (int i = 0; i < kSize; ++i) {
    if (v[i].value != snapshot_v[i].value || v[i].value != snapshot_custom_v[i].value) {
      all_equal = false;
    }
    if (i < kSize / 2) {
      assert(v[i].value <= v[kSize / 2].value);
    }
  }
  assert(all_equal);
}

#if TEST_STD_VER > 17
constexpr bool test_constexpr() {
  std::array<MyType, 10> v;
  for (int i = 9; i >= 0; --i) {
    v[9 - i].value = i;
  }
  std::nth_element(v.begin(), v.begin() + 5, v.end());
  return std::is_partitioned(v.begin(), v.end(), [&](const MyType& m) { return m.value <= v[5].value; });
}
#endif

int main(int, char**) {
  test_randomization();
  test_same();
#if TEST_STD_VER > 17
  static_assert(test_constexpr(), "");
#endif
  return 0;
}

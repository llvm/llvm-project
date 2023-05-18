//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// Test std::sort stability randomization

// UNSUPPORTED: c++03
// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DEBUG_RANDOMIZE_UNSPECIFIED_STABILITY

#include <algorithm>
#include <array>
#include <cassert>
#include <functional>
#include <iterator>
#include <vector>

#include "test_macros.h"

struct EqualType {
  int value = 0;
  constexpr bool operator<(const EqualType&) const { return false; }
};

std::vector<EqualType> deterministic() {
  static constexpr int kSize = 100;
  std::vector<EqualType> v;
  v.resize(kSize);
  for (int i = 0; i < kSize; ++i) {
    v[i].value = kSize / 2 - i * (i % 2 ? -1 : 1);
  }
  std::less<EqualType> comp;
  std::__sort_dispatch<std::_ClassicAlgPolicy>(v.begin(), v.end(), comp);
  return v;
}

void test_randomization() {
  static constexpr int kSize = 100;
  std::vector<EqualType> v;
  v.resize(kSize);
  for (int i = 0; i < kSize; ++i) {
    v[i].value = kSize / 2 - i * (i % 2 ? -1 : 1);
  }
  auto deterministic_v = deterministic();
  std::sort(v.begin(), v.end());
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
  std::vector<EqualType> v;
  v.resize(kSize);
  for (int i = 0; i < kSize; ++i) {
    v[i].value = kSize / 2 - i * (i % 2 ? -1 : 1);
  }
  auto snapshot_v = v;
  auto snapshot_custom_v = v;
  std::sort(v.begin(), v.end());
  std::sort(snapshot_v.begin(), snapshot_v.end());
  std::sort(snapshot_custom_v.begin(), snapshot_custom_v.end(),
            [](const EqualType&, const EqualType&) { return false; });
  bool all_equal = true;
  for (int i = 0; i < kSize; ++i) {
    if (v[i].value != snapshot_v[i].value || v[i].value != snapshot_custom_v[i].value) {
      all_equal = false;
    }
  }
  assert(all_equal);
}

#if TEST_STD_VER > 17
constexpr bool test_constexpr() {
  std::array<EqualType, 10> v;
  for (int i = 9; i >= 0; --i) {
    v[9 - i].value = i;
  }
  std::sort(v.begin(), v.end());
  return std::is_sorted(v.begin(), v.end());
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

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Test std::unordered_{set,map,multiset,multimap} randomization

// UNSUPPORTED: c++03
// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DEBUG_RANDOMIZE_UNSPECIFIED_STABILITY

#include <unordered_set>
#include <unordered_map>
#include <cassert>
#include <vector>
#include <algorithm>

const int kSize = 128;

template <typename T, typename F>
T get_random(F get_value) {
  T v;
  v.reserve(kSize);
  for (int i = 0; i < kSize; ++i) {
    v.insert(get_value());
  }
  v.rehash(v.bucket_count() + 1);
  return v;
}

template <typename T, typename F>
T get_deterministic(F get_value) {
  T v;
  v.reserve(kSize);
  for (int i = 0; i < kSize; ++i) {
    v.insert(get_value());
  }
  return v;
}

template <typename T>
struct RemoveConst {
  using type = T;
};

template <typename T, typename U>
struct RemoveConst<std::pair<const T, U>> {
  using type = std::pair<T, U>;
};

template <typename T, typename F>
void test_randomization(F get_value) {
  T t1 = get_deterministic<T>(get_value), t2 = get_random<T>(get_value);

  // Convert pair<const K, V> to pair<K, V> so it can be sorted
  using U = typename RemoveConst<typename T::value_type>::type;

  std::vector<U> t1v(t1.begin(), t1.end()), t2v(t2.begin(), t2.end());

  assert(t1v != t2v);

  std::sort(t1v.begin(), t1v.end());
  std::sort(t2v.begin(), t2v.end());

  assert(t1v == t2v);
}

int main(int, char**) {
  int i = 0, j = 0;
  test_randomization<std::unordered_set<int>>([i]() mutable { return i++; });
  test_randomization<std::unordered_map<int, int>>([i, j]() mutable { return std::make_pair(i++, j++); });
  test_randomization<std::unordered_multiset<int>>([i]() mutable { return i++ % 32; });
  test_randomization<std::unordered_multimap<int, int>>([i, j]() mutable { return std::make_pair(i++ % 32, j++); });
  return 0;
}

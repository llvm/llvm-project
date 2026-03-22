//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20, c++23
// XFAIL: !has-64-bit-atomics

// floating-point-type fetch_max(floating-point-type, memory_order = memory_order::seq_cst) const noexcept;

#include <atomic>
#include <cassert>
#include <concepts>
#include <type_traits>
#include <utility>

#include "test_macros.h"
#include "atomic_helpers.h"
#include "../atomics.types.operations/atomics.types.operations.req/atomic_fetch_fmax_helper.h"

template <typename T>
concept has_fetch_max = requires(T const& t, typename T::value_type v, std::memory_order m) {
  t.fetch_max(v);
  t.fetch_max(v, m);
};

template <typename T>
void test_does_not_have_fetch_max() {
  static_assert(!has_fetch_max<std::atomic_ref<T>>);
}

template <typename T>
struct TestFetchMax {
  void operator()() const {
    if constexpr (std::is_floating_point_v<T>) {
      alignas(std::atomic_ref<T>::required_alignment) T x;
      std::atomic_ref<T> const a(x);

      auto load   = [&]() { return x; };
      auto store  = [&](T val) { x = val; };
      auto max_op = [&](T val, auto order) { return a.fetch_max(val, order); };

      ASSERT_NOEXCEPT(a.fetch_max(T(0), std::memory_order_seq_cst));
      test_fetch_fmax<T>(load, store, max_op);

    } else {
      static_assert(std::is_void_v<T>);
    }
  }
};

int main(int, char**) {
  TestFetchMax<float>{}();
  TestFetchMax<double>{}();
  TestFetchMax<long double>{}();

  test_does_not_have_fetch_max<bool>();
  test_does_not_have_fetch_max<int>();
  test_does_not_have_fetch_max<unsigned int>();
  test_does_not_have_fetch_max<int*>();

  return 0;
}

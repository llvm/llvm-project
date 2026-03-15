//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20, c++23
// XFAIL: !has-64-bit-atomics

// floating-point-type fetch_fminimum_num(floating-point-type, memory_order = memory_order::seq_cst) const noexcept;

#include <atomic>
#include <cassert>
#include <concepts>
#include <type_traits>
#include <utility>

#include "test_macros.h"
#include "atomic_helpers.h"
#include "../atomics.types.operations/atomics.types.operations.req/atomic_fetch_fminimum_num_helper.h"

template <typename T>
concept has_fetch_fminimum_num = requires(T t) {
  std::declval<T const>().fetch_fminimum_num(std::declval<typename T::value_type>());
  std::declval<T const>().fetch_fminimum_num(std::declval<typename T::value_type>(), std::declval<std::memory_order>());
};

template <typename T>
struct TestDoesNotHaveFetchFMinimumNum {
  void operator()() const { static_assert(!has_fetch_fminimum_num<std::atomic_ref<T>>); }
};

template <typename T>
struct TestFetchFMinimumNum {
  void operator()() const {
    if constexpr (std::is_floating_point_v<T>) {
      alignas(std::atomic_ref<T>::required_alignment) T x;
      std::atomic_ref<T> const a(x);

      auto load         = [&]() { return x; };
      auto store        = [&](T val) { x = val; };
      auto fminimum_num = [&](T val, auto order) { return a.fetch_fminimum_num(val, order); };

      ASSERT_NOEXCEPT(a.fetch_fminimum_num(T(0), std::memory_order_seq_cst));
      test_fetch_fminimum_num<T>(load, store, fminimum_num);

    } else {
      static_assert(std::is_void_v<T>);
    }
  }
};

int main(int, char**) {
  TestFetchFMinimumNum<float>{}();
  TestFetchFMinimumNum<double>{}();

  TestDoesNotHaveFetchFMinimumNum<bool>{}();
  TestDoesNotHaveFetchFMinimumNum<int>{}();
  TestDoesNotHaveFetchFMinimumNum<unsigned int>{}();
  TestDoesNotHaveFetchFMinimumNum<int*>{}();

  return 0;
}

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// class sequenced_policy;
// class parallel_policy;
// class parallel_unsequenced_policy;
// class unsequenced_policy; // since C++20
//
// inline constexpr sequenced_policy seq = implementation-defined;
// inline constexpr parallel_policy par = implementation-defined;
// inline constexpr parallel_unsequenced_policy par_unseq = implementation-defined;
// inline constexpr unsequenced_policy unseq = implementation-defined; // since C++20

// UNSUPPORTED: c++03, c++11, c++14

// UNSUPPORTED: libcpp-has-no-incomplete-pstl

#include <execution>
#include <type_traits>

#include "test_macros.h"

template <class T>
using remove_cvref_t = std::remove_cv_t<std::remove_reference_t<T>>;

template <class T>
TEST_NOINLINE void use(T&) {}

static_assert(std::is_same_v<remove_cvref_t<decltype(std::execution::seq)>, std::execution::sequenced_policy>);
static_assert(std::is_same_v<remove_cvref_t<decltype(std::execution::par)>, std::execution::parallel_policy>);
static_assert(
    std::is_same_v<remove_cvref_t<decltype(std::execution::par_unseq)>, std::execution::parallel_unsequenced_policy>);

#if TEST_STD_VER >= 20
static_assert(std::is_same_v<remove_cvref_t<decltype(std::execution::unseq)>, std::execution::unsequenced_policy>);
#endif

int main(int, char**) {
  use(std::execution::seq);
  use(std::execution::par);
  use(std::execution::par_unseq);
#if TEST_STD_VER >= 20
  use(std::execution::unseq);
#endif

  return 0;
}

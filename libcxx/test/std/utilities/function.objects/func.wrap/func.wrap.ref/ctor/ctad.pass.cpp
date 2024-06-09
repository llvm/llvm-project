//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20, c++23

#include <cassert>
#include <functional>
#include <utility>
#include <type_traits>

#include "test_macros.h"

int fn(int, float) { return 42; }

int fn_noexcept(int, float) noexcept { return 42; }

struct S {
  int data_mem = 42;

  int fn_mem(int, float) { return 42; }
  int fn_mem_noexcept(int, float) noexcept { return 42; }
};

void test() {
  // template<class F>
  //  function_ref(F*) -> function_ref<F>;
  {
    std::function_ref fn_ref = fn;
    static_assert(std::is_same_v<decltype(fn_ref), std::function_ref<int(int, float)>>);
  }
  {
    std::function_ref fn_ref = fn_noexcept;
    static_assert(std::is_same_v<decltype(fn_ref), std::function_ref<int(int, float) noexcept>>);
  }

  // template<auto f>
  //  function_ref(nontype_t<f>) -> function_ref<...>;
  {
    std::function_ref fn_ref = std::nontype_t<fn>();
    static_assert(std::is_same_v<decltype(fn_ref), std::function_ref<int(int, float)>>);
  }
  {
    std::function_ref fn_ref = std::nontype_t<fn_noexcept>();
    static_assert(std::is_same_v<decltype(fn_ref), std::function_ref<int(int, float) noexcept>>);
  }

  // template<auto f, class T>
  //  function_ref(nontype_t<f>, T&&) -> function_ref<...>;
  {
    int arg                  = 0;
    std::function_ref fn_ref = {std::nontype_t<fn>(), arg};
    static_assert(std::is_same_v<decltype(fn_ref), std::function_ref<int(float)>>);
  }
  {
    S s;
    std::function_ref fn_ref = {std::nontype_t<&S::data_mem>(), s};
    static_assert(std::is_same_v<decltype(fn_ref), std::function_ref<int&()>>);
  }
  {
    const S s;
    std::function_ref fn_ref = {std::nontype_t<&S::data_mem>(), s};
    static_assert(std::is_same_v<decltype(fn_ref), std::function_ref<int const&()>>);
  }
  {
    S s;
    std::function_ref fn_ref = {std::nontype_t<&S::fn_mem>(), s};
    static_assert(std::is_same_v<decltype(fn_ref), std::function_ref<int(int, float)>>);
  }
  {
    S s;
    std::function_ref fn_ref = {std::nontype_t<&S::fn_mem_noexcept>(), s};
    static_assert(std::is_same_v<decltype(fn_ref), std::function_ref<int(int, float) noexcept>>);
  }
}

int main(int, char**) {
  test();
  return 0;
}

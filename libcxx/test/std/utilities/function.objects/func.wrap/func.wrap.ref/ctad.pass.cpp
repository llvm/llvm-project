//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

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
  int fn_mem_ref(int, float) & { return 42; }
  int fn_mem_const(int, float) const { return 42; }
  int fn_mem_const_ref(int, float) const& { return 42; }
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

  // template<auto c, class F>
  //  function_ref(constant_wrapper<c, F>) -> function_ref<...>;
  {
    std::function_ref fn_ref = std::constant_wrapper<fn>();
    static_assert(std::is_same_v<decltype(fn_ref), std::function_ref<int(int, float)>>);
  }
  {
    std::function_ref fn_ref = std::constant_wrapper<fn_noexcept>();
    static_assert(std::is_same_v<decltype(fn_ref), std::function_ref<int(int, float) noexcept>>);
  }

  // template<auto c, class F, class T>
  //  function_ref(constant_wrapper<c, F>, T&&) -> function_ref<...>;
  {
    int arg                  = 0;
    std::function_ref fn_ref = {std::constant_wrapper<fn>(), arg};
    static_assert(std::is_same_v<decltype(fn_ref), std::function_ref<int(float)>>);
  }
  {
    int arg                  = 0;
    std::function_ref fn_ref = {std::constant_wrapper<fn_noexcept>(), arg};
    static_assert(std::is_same_v<decltype(fn_ref), std::function_ref<int(float) noexcept>>);
  }
  {
    S s;
    std::function_ref fn_ref = {std::constant_wrapper<&S::data_mem>(), s};
    static_assert(std::is_same_v<decltype(fn_ref), std::function_ref<int&() noexcept>>);
  }
  {
    const S s;
    std::function_ref fn_ref = {std::constant_wrapper<&S::data_mem>(), s};
    static_assert(std::is_same_v<decltype(fn_ref), std::function_ref<int const&() noexcept>>);
  }
  {
    S s;
    std::function_ref fn_ref = {std::constant_wrapper<&S::fn_mem>(), s};
    static_assert(std::is_same_v<decltype(fn_ref), std::function_ref<int(int, float)>>);
  }
  {
    S s;
    std::function_ref fn_ref = {std::constant_wrapper<&S::fn_mem_const>(), s};
    static_assert(std::is_same_v<decltype(fn_ref), std::function_ref<int(int, float)>>);
  }
  {
    S s;
    std::function_ref fn_ref = {std::constant_wrapper<&S::fn_mem_ref>(), s};
    static_assert(std::is_same_v<decltype(fn_ref), std::function_ref<int(int, float)>>);
  }
  {
    S s;
    std::function_ref fn_ref = {std::constant_wrapper<&S::fn_mem_const_ref>(), s};
    static_assert(std::is_same_v<decltype(fn_ref), std::function_ref<int(int, float)>>);
  }
  {
    S s;
    std::function_ref fn_ref = {std::constant_wrapper<&S::fn_mem_noexcept>(), s};
    static_assert(std::is_same_v<decltype(fn_ref), std::function_ref<int(int, float) noexcept>>);
  }
}

int main(int, char**) {
  test();
  return 0;
}

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

static_assert(std::is_invocable_v<std::function_ref<void() noexcept>&>);
static_assert(std::is_invocable_v<std::function_ref<void() noexcept>>);
static_assert(std::is_invocable_v<std::function_ref<void() noexcept>&&>);
static_assert(std::is_invocable_v<std::function_ref<void() noexcept> const&>);
static_assert(std::is_invocable_v<std::function_ref<void() noexcept> const>);
static_assert(std::is_invocable_v<std::function_ref<void() noexcept> const&&>);

int fn() noexcept { return 42; }

struct {
  int operator()() noexcept { return 42; }
} fn_obj;

void test() {
  // template<class F> function_ref(F* f) noexcept;
  {
    // initialized from a function
    std::function_ref<int() noexcept> fn_ref = fn;
    assert(fn_ref() == 42);
  }
  {
    // initialized from a function pointer
    std::function_ref<int() noexcept> fn_ref = &fn;
    assert(fn_ref() == 42);
  }

  // template<class F> constexpr function_ref(F&& f) noexcept;
  {
    // initialized from a function object
    std::function_ref<int() noexcept> fn_ref = fn_obj;
    assert(fn_ref() == 42);
  }
}

struct S {
  int data_mem = 42;

  int fn_mem() noexcept { return 42; }
};

void test_nontype_t() {
  // template<auto f> constexpr function_ref(nontype_t<f>) noexcept;
  {
    // initialized from a function through `nontype_t`
    std::function_ref<int() noexcept> fn_ref = std::nontype_t<fn>();
    assert(fn_ref() == 42);
  }
  {
    // initialized from a function pointer through `nontype_t`
    std::function_ref<int() noexcept> fn_ref = std::nontype_t<&fn>();
    assert(fn_ref() == 42);
  }
  {
    S s;
    // initialized from a pointer to data member through `nontype_t`
    std::function_ref<int(S) noexcept> fn_ref = std::nontype_t<&S::data_mem>();
    assert(fn_ref(s) == 42);
  }
  {
    S s;
    // initialized from a pointer to function member through `nontype_t`
    std::function_ref<int(S) noexcept> fn_ref = std::nontype_t<&S::fn_mem>();
    assert(fn_ref(s) == 42);
  }

  // template<auto f, class U>
  //   constexpr function_ref(nontype_t<f>, U&& obj) noexcept;
  {
    S s;
    // initialized from a pointer to data member through `nontype_t` and bound to an object through a reference
    std::function_ref<int() noexcept> fn_ref = {std::nontype_t<&S::data_mem>(), s};
    assert(fn_ref() == 42);
  }
  {
    S s;
    // initialized from a pointer to function member through `nontype_t` and bound to an object through a reference
    std::function_ref<int() noexcept> fn_ref = {std::nontype_t<&S::fn_mem>(), s};
    assert(fn_ref() == 42);
  }

  // template<auto f, class T>
  //   constexpr function_ref(nontype_t<f>, cv T* obj) noexcept;
  {
    S s;
    // initialized from a pointer to data member through `nontype_t` and bound to an object through a pointer
    std::function_ref<int() noexcept> fn_ref = {std::nontype_t<&S::data_mem>(), &s};
    assert(fn_ref() == 42);
  }
  {
    S s;
    // initialized from a pointer to function member through `nontype_t` and bound to an object through a pointer
    static_assert(std::is_same_v<decltype(&s), S*>);
    std::function_ref<int() noexcept> fn_ref = {std::nontype_t<&S::fn_mem>(), &s};
    assert(fn_ref() == 42);
  }
}

int main(int, char**) {
  test();
  test_nontype_t();
  return 0;
}

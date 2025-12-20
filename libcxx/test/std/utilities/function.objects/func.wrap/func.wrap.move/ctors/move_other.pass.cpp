//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// GCC doesn't support [[clang::trivial_abi]] currently, which we want to use on
// move_only_function.
// UNSUPPORTED: gcc

#include <cassert>
#include <functional>
#include <utility>

#include "type_algorithms.h"

static_assert(!std::is_constructible_v<std::move_only_function<void() const>, std::move_only_function<void()>>);
static_assert(!std::is_constructible_v<std::move_only_function<void() noexcept>, std::move_only_function<void()>>);
static_assert(
    !std::is_constructible_v<std::move_only_function<void() const noexcept>, std::move_only_function<void()>>);
static_assert(
    !std::is_constructible_v<std::move_only_function<void() const noexcept>, std::move_only_function<void() const>>);
static_assert(
    !std::is_constructible_v<std::move_only_function<void() const noexcept>, std::move_only_function<void() noexcept>>);

template <class T>
void test() {
  {
    std::move_only_function<void() const noexcept> f1;
    std::move_only_function<T> f2 = std::move(f1);
    assert(!f2);
  }
  {
    std::move_only_function<void() const & noexcept> f1;
    std::move_only_function<T> f2 = std::move(f1);
    assert(!f2);
  }
}

int main(int, char**) {
  types::for_each(types::function_noexcept_const_ref_qualified<void()>{}, []<class T> { test<T>(); });

  return 0;
}

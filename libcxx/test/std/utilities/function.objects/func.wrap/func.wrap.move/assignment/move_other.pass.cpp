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
#include <concepts>
#include <functional>
#include <utility>

#include "type_algorithms.h"
#include "../common.h"

template <class T>
void test() {
  {
    std::move_only_function<void() const noexcept> f1;
    std::move_only_function<T> f2;
    std::same_as<std::move_only_function<T>&> decltype(auto) ret = (f2 = std::move(f1));
    assert(&ret == &f2);
    assert(!f2);
  }
  {
    std::move_only_function<void() const & noexcept> f1;
    std::move_only_function<T> f2;
    std::same_as<std::move_only_function<T>&> decltype(auto) ret = (f2 = std::move(f1));
    assert(&ret == &f2);
    assert(!f2);
  }
}

template <class T>
void test2() {
  {
    std::move_only_function<int() const noexcept> f1 = [] noexcept { return 109; };
    std::move_only_function<T> f2;
    std::same_as<std::move_only_function<T>&> decltype(auto) ret = (f2 = std::move(f1));
    assert(&ret == &f2);
    assert(f2);
    assert(f2() == 109);
  }
  {
    std::move_only_function<int() const& noexcept> f1 = [] noexcept { return 109; };
    std::move_only_function<T> f2;
    std::same_as<std::move_only_function<T>&> decltype(auto) ret = (f2 = std::move(f1));
    assert(&ret == &f2);
    assert(f2);
    assert(f2() == 109);
  }
}

int main(int, char**) {
  types::for_each(types::function_noexcept_const_ref_qualified<void()>{}, []<class T> { test<T>(); });
  types::for_each(types::function_noexcept_const_lvalue_ref_qualified<int()>{}, []<class T> { test2<T>(); });

  return 0;
}

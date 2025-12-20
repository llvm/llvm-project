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

#include "test_macros.h"
#include "type_algorithms.h"
#include "../common.h"

template <class T>
void test() {
  {
    int counter = 0;
    std::move_only_function<T> f{std::in_place_type<TriviallyDestructible>, MoveCounter{&counter}};
    assert(f);
    assert(counter == 1);
  }
  {
    int counter = 0;
    std::move_only_function<T> f{std::in_place_type<TriviallyDestructibleTooLarge>, MoveCounter{&counter}};
    assert(f);
    assert(counter == 1);
  }
  {
    int counter = 0;
    std::move_only_function<T> f{std::in_place_type<NonTrivial>, MoveCounter{&counter}};
    assert(f);
    assert(counter == 1);
  }
}

int main(int, char**) {
  types::for_each(types::function_noexcept_const_ref_qualified<void()>{}, []<class T> { test<T>(); });
  types::for_each(types::function_noexcept_const_ref_qualified<int(int)>{}, []<class T> { test<T>(); });

  return 0;
}

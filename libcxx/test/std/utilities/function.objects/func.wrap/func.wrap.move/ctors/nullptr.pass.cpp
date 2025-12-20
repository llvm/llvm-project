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

#include "type_algorithms.h"

template <class T>
void test() {
  std::move_only_function<T> f = nullptr;
  assert(!f);
}

int main(int, char**) {
  types::for_each(types::function_noexcept_const_ref_qualified<void()>{}, []<class T> { test<T>(); });
  types::for_each(types::function_noexcept_const_ref_qualified<int()>{}, []<class T> { test<T>(); });
  types::for_each(types::function_noexcept_const_ref_qualified<int(int)>{}, []<class T> { test<T>(); });

  return 0;
}

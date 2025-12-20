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
#include "common.h"

template <class T>
void test() {
  {
    std::move_only_function<T> f = &call_func;
    std::move_only_function<T> f2;
    swap(f, f2);
  }
  {
    decltype(&call_func) ptr     = nullptr;
    std::move_only_function<T> f = ptr;
    std::move_only_function<T> f2;
    swap(f, f2);
  }
  {
    std::move_only_function<T> f = TriviallyDestructible{};
    std::move_only_function<T> f2;
    swap(f, f2);
  }
  {
    std::move_only_function<T> f = TriviallyDestructibleTooLarge{};
    std::move_only_function<T> f2;
    swap(f, f2);
  }
  {
    std::move_only_function<T> f = NonTrivial{};
    std::move_only_function<T> f2;
    swap(f, f2);
  }
}

struct S {
  void func() noexcept {}
};

template <class T>
void test_member_function_pointer() {
  {
    std::move_only_function<T> f = &S::func;
    std::move_only_function<T> f2;
    swap(f, f2);
  }
  {
    decltype(&S::func) ptr       = nullptr;
    std::move_only_function<T> f = ptr;
    std::move_only_function<T> f2;
    swap(f, f2);
  }
}

int main(int, char**) {
  types::for_each(types::function_noexcept_const_ref_qualified<void()>{}, []<class T> { test<T>(); });
  types::for_each(types::function_noexcept_const_ref_qualified<void(S)>{}, []<class T> {
    test_member_function_pointer<T>();
  });

  return 0;
}

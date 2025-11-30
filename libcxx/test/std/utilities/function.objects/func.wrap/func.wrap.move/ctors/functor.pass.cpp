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

#include "count_new.h"
#include "test_macros.h"
#include "type_algorithms.h"
#include "../common.h"

template <class T>
void test() {
  {
    std::move_only_function<T> f = &call_func;
    assert(f);
  }
  {
    decltype(&call_func) ptr     = nullptr;
    std::move_only_function<T> f = ptr;
    assert(!f);
  }
  {
    std::move_only_function<T> f = TriviallyDestructible{};
    assert(f);
  }
  {
    std::move_only_function<T> f = TriviallyDestructibleTooLarge{};
    assert(f);
  }
  {
    std::move_only_function<T> f = NonTrivial{};
    assert(f);
  }
}

struct S {
  void func() noexcept {}
};

template <class T>
void test_member_function_pointer() {
  {
    std::move_only_function<T> f = &S::func;
    assert(f);
  }
  {
    decltype(&S::func) ptr       = nullptr;
    std::move_only_function<T> f = ptr;
    assert(!f);
  }
}

template <class T>
void test_value_return_type() {
  {
    std::move_only_function<T> f = &get_val;
    assert(f);
  }
  {
    decltype(&get_val) ptr       = nullptr;
    std::move_only_function<T> f = ptr;
    assert(!f);
  }
  {
    std::move_only_function<T> f = TriviallyDestructible{};
    assert(f);
  }
  {
    std::move_only_function<T> f = TriviallyDestructibleTooLarge{};
    assert(f);
  }
  {
    std::move_only_function<T> f = NonTrivial{};
    assert(f);
  }
}

template <class T>
void test_throwing() {
  struct ThrowingFunctor {
    ThrowingFunctor() = default;
    ThrowingFunctor(const ThrowingFunctor&) { throw 1; }
    void operator()() {}
  };
  std::move_only_function<T> func({});
}

void check_new_delete_called() {
  assert(globalMemCounter.new_called == globalMemCounter.delete_called);
  assert(globalMemCounter.new_array_called == globalMemCounter.delete_array_called);
  assert(globalMemCounter.aligned_new_called == globalMemCounter.aligned_delete_called);
  assert(globalMemCounter.aligned_new_array_called == globalMemCounter.aligned_delete_array_called);
}

int main(int, char**) {
  types::for_each(types::function_noexcept_const_ref_qualified<void()>{}, []<class T> { test<T>(); });
  types::for_each(types::function_noexcept_const_ref_qualified<void(S)>{}, []<class T> {
    test_member_function_pointer<T>();
  });
  types::for_each(types::function_noexcept_const_ref_qualified<int(int)>{}, []<class T> {
    test_value_return_type<T>();
  });
  types::for_each(types::function_noexcept_const_ref_qualified<void()>{}, []<class T> { test_throwing<T>(); });
  check_new_delete_called();

  return 0;
}

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

#include <cassert>
#include <functional>

#include "test_macros.h"
#include "../common.h"

template <class T>
void test() {
  {
    std::move_only_function<T> f = &call_func;
    f = nullptr;
    assert(!f);
  }
  {
    decltype(&call_func) ptr     = nullptr;
    std::move_only_function<T> f = ptr;
    f = nullptr;
    assert(!f);
  }
  {
    std::move_only_function<T> f = TriviallyDestructible{};
    f = nullptr;
    assert(!f);
  }
  {
    std::move_only_function<T> f = TriviallyDestructibleSqueezeFit{};
    f = nullptr;
    assert(!f);
  }
  {
    std::move_only_function<T> f = TriviallyDestructibleTooLarge{};
    f = nullptr;
    assert(!f);
  }
#ifdef TEST_COMPILER_CLANG
  {
    std::move_only_function<T> f = TriviallyRelocatable{};
    f = nullptr;
    assert(!f);
  }
  {
    std::move_only_function<T> f = TriviallyRelocatableSqueezeFit{};
    f = nullptr;
    assert(!f);
  }
  {
    std::move_only_function<T> f = TriviallyRelocatableTooLarge{};
    f = nullptr;
    assert(!f);
  }
#endif
  {
    std::move_only_function<T> f = NonTrivial{};
    f = nullptr;
    assert(!f);
  }
}

struct S {
  void func() noexcept {}
};

template <class T>
void test_member_function_pointer() {
  {
    std::move_only_function<T> f = &S::func;
    f = nullptr;
    assert(!f);
  }
  {
    decltype(&S::func) ptr       = nullptr;
    std::move_only_function<T> f = ptr;
    f = nullptr;
    assert(!f);
  }
}

int main(int, char**) {
  call_test<void()>([]<class T> { test<T>(); });
  call_test<void(S)>([]<class T> { test_member_function_pointer<T>(); });
}

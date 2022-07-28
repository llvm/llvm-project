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
    assert(f);
    LIBCPP_ASSERT(f.__get_status() == std::__move_only_function_storage::_Status::_TriviallyDestructible);
  }
  {
    decltype(&call_func) ptr     = nullptr;
    std::move_only_function<T> f = ptr;
    assert(!f);
    LIBCPP_ASSERT(f.__get_status() == std::__move_only_function_storage::_Status::_NotEngaged);
  }
  {
    std::move_only_function<T> f = TriviallyDestructible{};
    assert(f);
    LIBCPP_ASSERT(f.__get_status() == std::__move_only_function_storage::_Status::_TriviallyDestructible);
  }
  {
    std::move_only_function<T> f = TriviallyDestructibleSqueezeFit{};
    assert(f);
    LIBCPP_ASSERT(f.__get_status() == std::__move_only_function_storage::_Status::_TriviallyDestructible);
  }
  {
    std::move_only_function<T> f = TriviallyDestructibleTooLarge{};
    assert(f);
    LIBCPP_ASSERT(f.__get_status() == std::__move_only_function_storage::_Status::_Heap);
  }
#ifdef TEST_COMPILER_CLANG
  {
    std::move_only_function<T> f = TriviallyRelocatable{};
    assert(f);
    LIBCPP_ASSERT(f.__get_status() == std::__move_only_function_storage::_Status::_TriviallyRelocatable);
  }
  {
    std::move_only_function<T> f = TriviallyRelocatableSqueezeFit{};
    assert(f);
    LIBCPP_ASSERT(f.__get_status() == std::__move_only_function_storage::_Status::_TriviallyRelocatable);
  }
  {
    std::move_only_function<T> f = TriviallyRelocatableTooLarge{};
    assert(f);
    LIBCPP_ASSERT(f.__get_status() == std::__move_only_function_storage::_Status::_Heap);
  }
#endif
  {
    std::move_only_function<T> f = NonTrivial{};
    assert(f);
    LIBCPP_ASSERT(f.__get_status() == std::__move_only_function_storage::_Status::_Heap);
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
    LIBCPP_ASSERT(f.__get_status() == std::__move_only_function_storage::_Status::_TriviallyDestructible);
  }
  {
    decltype(&S::func) ptr       = nullptr;
    std::move_only_function<T> f = ptr;
    assert(!f);
    LIBCPP_ASSERT(f.__get_status() == std::__move_only_function_storage::_Status::_NotEngaged);
  }
}

int main(int, char**) {
  call_test<void()>([]<class T> { test<T>(); });
  call_test<void(S)>([]<class T> { test_member_function_pointer<T>(); });
}

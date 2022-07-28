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
#include "common.h"

template <class T>
void test() {
  {
    std::move_only_function<T> f = &call_func;
    std::move_only_function<T> f2;
    f.swap(f2);
    LIBCPP_ASSERT(f.__get_status() == std::__move_only_function_storage::_Status::_NotEngaged);
    LIBCPP_ASSERT(f2.__get_status() == std::__move_only_function_storage::_Status::_TriviallyDestructible);
  }
  {
    decltype(&call_func) ptr     = nullptr;
    std::move_only_function<T> f = ptr;
    std::move_only_function<T> f2;
    f.swap(f2);
    LIBCPP_ASSERT(f.__get_status() == std::__move_only_function_storage::_Status::_NotEngaged);
    LIBCPP_ASSERT(f2.__get_status() == std::__move_only_function_storage::_Status::_NotEngaged);
  }
  {
    std::move_only_function<T> f = TriviallyDestructible{};
    std::move_only_function<T> f2;
    f.swap(f2);
    LIBCPP_ASSERT(f.__get_status() == std::__move_only_function_storage::_Status::_NotEngaged);
    LIBCPP_ASSERT(f2.__get_status() == std::__move_only_function_storage::_Status::_TriviallyDestructible);
  }
  {
    std::move_only_function<T> f = TriviallyDestructibleSqueezeFit{};
    std::move_only_function<T> f2;
    f.swap(f2);
    LIBCPP_ASSERT(f.__get_status() == std::__move_only_function_storage::_Status::_NotEngaged);
    LIBCPP_ASSERT(f2.__get_status() == std::__move_only_function_storage::_Status::_TriviallyDestructible);
  }
  {
    std::move_only_function<T> f = TriviallyDestructibleTooLarge{};
    std::move_only_function<T> f2;
    f.swap(f2);
    LIBCPP_ASSERT(f.__get_status() == std::__move_only_function_storage::_Status::_NotEngaged);
    LIBCPP_ASSERT(f2.__get_status() == std::__move_only_function_storage::_Status::_Heap);
  }
#ifdef TEST_COMPILER_CLANG
  {
    std::move_only_function<T> f = TriviallyRelocatable{};
    std::move_only_function<T> f2;
    f.swap(f2);
    LIBCPP_ASSERT(f.__get_status() == std::__move_only_function_storage::_Status::_NotEngaged);
    LIBCPP_ASSERT(f2.__get_status() == std::__move_only_function_storage::_Status::_TriviallyRelocatable);
  }
  {
    std::move_only_function<T> f = TriviallyRelocatableSqueezeFit{};
    std::move_only_function<T> f2;
    f.swap(f2);
    LIBCPP_ASSERT(f.__get_status() == std::__move_only_function_storage::_Status::_NotEngaged);
    LIBCPP_ASSERT(f2.__get_status() == std::__move_only_function_storage::_Status::_TriviallyRelocatable);
  }
  {
    std::move_only_function<T> f = TriviallyRelocatableTooLarge{};
    std::move_only_function<T> f2;
    f.swap(f2);
    LIBCPP_ASSERT(f.__get_status() == std::__move_only_function_storage::_Status::_NotEngaged);
    LIBCPP_ASSERT(f2.__get_status() == std::__move_only_function_storage::_Status::_Heap);
  }
#endif
  {
    std::move_only_function<T> f = NonTrivial{};
    std::move_only_function<T> f2;
    f.swap(f2);
    LIBCPP_ASSERT(f.__get_status() == std::__move_only_function_storage::_Status::_NotEngaged);
    LIBCPP_ASSERT(f2.__get_status() == std::__move_only_function_storage::_Status::_Heap);
  }
  {
    std::move_only_function<T> f = TriviallyDestructibleSqueezeFit{};
    std::move_only_function<T> f2 = NonTrivial{};
    f.swap(f2);
    LIBCPP_ASSERT(f.__get_status() == std::__move_only_function_storage::_Status::_Heap);
    LIBCPP_ASSERT(f2.__get_status() == std::__move_only_function_storage::_Status::_TriviallyDestructible);
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
    f.swap(f2);
    LIBCPP_ASSERT(f.__get_status() == std::__move_only_function_storage::_Status::_NotEngaged);
    LIBCPP_ASSERT(f2.__get_status() == std::__move_only_function_storage::_Status::_TriviallyDestructible);
  }
  {
    decltype(&S::func) ptr       = nullptr;
    std::move_only_function<T> f = ptr;
    std::move_only_function<T> f2;
    f.swap(f2);
    LIBCPP_ASSERT(f.__get_status() == std::__move_only_function_storage::_Status::_NotEngaged);
    LIBCPP_ASSERT(f2.__get_status() == std::__move_only_function_storage::_Status::_NotEngaged);
  }
}

int main(int, char**) {
  call_test<void()>([]<class T> { test<T>(); });
  call_test<void(S)>([]<class T> { test_member_function_pointer<T>(); });
}

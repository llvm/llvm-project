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
    int counter = 0;
    std::move_only_function<T> f{std::in_place_type<TriviallyDestructible>, MoveCounter{&counter}};
    assert(f);
    assert(counter == 1);
    LIBCPP_ASSERT(f.__get_status() == std::__move_only_function_storage::_Status::_TriviallyDestructible);
  }
  {
    int counter = 0;
    std::move_only_function<T> f{std::in_place_type<TriviallyDestructibleSqueezeFit>, MoveCounter{&counter}};
    assert(f);
    assert(counter == 1);
    LIBCPP_ASSERT(f.__get_status() == std::__move_only_function_storage::_Status::_TriviallyDestructible);
  }
  {
    int counter = 0;
    std::move_only_function<T> f{std::in_place_type<TriviallyDestructibleTooLarge>, MoveCounter{&counter}};
    assert(f);
    assert(counter == 1);
    LIBCPP_ASSERT(f.__get_status() == std::__move_only_function_storage::_Status::_Heap);
  }
#ifdef TEST_COMPILER_CLANG
  {
    int counter = 0;
    std::move_only_function<T> f{std::in_place_type<TriviallyRelocatable>, MoveCounter{&counter}};
    assert(f);
    assert(counter == 1);
    LIBCPP_ASSERT(f.__get_status() == std::__move_only_function_storage::_Status::_TriviallyRelocatable);
  }
  {
    int counter = 0;
    std::move_only_function<T> f{std::in_place_type<TriviallyRelocatableSqueezeFit>, MoveCounter{&counter}};
    assert(f);
    assert(counter == 1);
    LIBCPP_ASSERT(f.__get_status() == std::__move_only_function_storage::_Status::_TriviallyRelocatable);
  }
  {
    int counter = 0;
    std::move_only_function<T> f{std::in_place_type<TriviallyRelocatableTooLarge>, MoveCounter{&counter}};
    assert(f);
    assert(counter == 1);
    LIBCPP_ASSERT(f.__get_status() == std::__move_only_function_storage::_Status::_Heap);
  }
#endif
  {
    int counter = 0;
    std::move_only_function<T> f{std::in_place_type<NonTrivial>, MoveCounter{&counter}};
    assert(f);
    assert(counter == 1);
    LIBCPP_ASSERT(f.__get_status() == std::__move_only_function_storage::_Status::_Heap);
  }
}

int main(int, char**) {
  call_test<void()>([]<class T> { test<T>(); });
}

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <memory>

// [inout.ptr.t], class template inout_ptr_t
// template<class Smart, class Pointer, class... Args>
//   class inout_ptr_t;                                        // since c++23

// explicit inout_ptr_t(Smart&, Args...);

#include <cassert>
#include <memory>

#include "test_convertible.h"
#include "../types.h"

int main(int, char**) {
  {
    std::unique_ptr<int> uPtr;

    std::inout_ptr_t<std::unique_ptr<int>, int*>{uPtr};

    static_assert(
        !test_convertible<std::inout_ptr_t<std::unique_ptr<int>, int*>>(), "This constructor must be explicit");

    // Test the state of the pointer after construction. Complete tests are available in inout_ptr.general.pass.cpp.
    assert(uPtr == nullptr);
  }

  {
    auto deleter = [](auto* p) { delete p; };
    std::unique_ptr<int, decltype(deleter)> uPtr;

    std::inout_ptr_t<std::unique_ptr<int, decltype(deleter)>, int*>{uPtr};

    static_assert(!test_convertible<std::inout_ptr_t<std::unique_ptr<int, decltype(deleter)>, int*>>(),
                  "This constructor must be explicit");

    // Test the state of the pointer after construction. Complete tests are available in inout_ptr.general.pass.cpp.
    assert(uPtr == nullptr);
  }

  {
    std::unique_ptr<int, MoveOnlyDeleter<int>> uPtr;

    std::inout_ptr_t<decltype(uPtr), int*, MoveOnlyDeleter<int>>{uPtr, MoveOnlyDeleter<int>{}};

    // Test the state of the pointer after construction. Complete tests are available in inout_ptr.general.pass.cpp.
    assert(uPtr == nullptr);
  }

  return 0;
}

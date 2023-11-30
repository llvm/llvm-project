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

#include <memory>

#include "test_convertible.h"

int main(int, char**) {
  {
    std::unique_ptr<int> uPtr;

    std::inout_ptr_t<std::unique_ptr<int>, int*>{uPtr};

    static_assert(
        !test_convertible<std::inout_ptr_t<std::unique_ptr<int>, int*>>(), "This constructor must be explicit");
  }
  {
    auto deleter = [](auto* p) { delete p; };
    std::unique_ptr<int, decltype(deleter)> uPtr;

    std::inout_ptr_t<std::unique_ptr<int, decltype(deleter)>, int*>{uPtr};

    static_assert(!test_convertible<std::inout_ptr_t<std::unique_ptr<int, decltype(deleter)>, int*>>(),
                  "This constructor must be explicit");
  }

  return 0;
}

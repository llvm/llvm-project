//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <memory>

// [inout.ptr], function template inout_ptr
// template<class Pointer = void, class Smart, class... Args>
//   auto inout_ptr(Smart& s, Args&&... args);                 // since c++23

#include <memory>
#include <tuple>

#include "../types.h"

int main(int, char**) {
  // `std::inout_ptr<>` does not support `std::shared_ptr<>`.
  {
    std::shared_ptr<int> sPtr;

    // expected-error-re@*:* {{static assertion failed due to requirement '!__is_specialization_v<std::shared_ptr<int>, std::shared_ptr>'{{.*}}std::shared_ptr<> is not supported}}
    std::ignore = std::inout_ptr(sPtr);
    // expected-error@*:* {{no matching conversion for functional-style cast from 'std::shared_ptr<int>' to 'std::inout_ptr_t<shared_ptr<int>, _Ptr>' (aka 'inout_ptr_t<std::shared_ptr<int>, int *>'}}
    std::ignore = std::inout_ptr<int*>(sPtr);
  }

#if 0

  {
    // CopyableMovableDeleter<int> del;
    // std::unique_ptr<int> uPtr;

    // -expected-error@*:* {{static assertion failed due to requirement 'is_constructible_v<std::unique_ptr<int, std::default_delete<int>>, int *, CopyableMovableDeleter<int> &>'}}
    // -expected-error@*:* {{no matching constructor for initialization of 'std::unique_ptr<int>'}}
    // std::ignore = std::inout_ptr(uPtr, del);
    // std::ignore = std::inout_ptr<int*>(uPtr, del);
  }

  {
    NotCopyAbleNotMovableDeleter<int> del;
    std::unique_ptr<int> uPtr;

    // e-xpected-error@*:* {{static assertion failed due to requirement 'is_constructible_v<std::unique_ptr<int, std::default_delete<int>>, int *, CopyableMovableDeleter<int> &>'}}
    std::ignore = std::inout_ptr(uPtr, del);
    std::ignore = std::inout_ptr<int*>(uPtr, del);
  }
#endif

  return 0;
}

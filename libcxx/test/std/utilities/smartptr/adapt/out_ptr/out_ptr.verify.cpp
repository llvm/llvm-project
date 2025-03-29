//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <memory>

// [out.ptr], function template out_ptr
// template<class Pointer = void, class Smart, class... Args>
//   auto out_ptr(Smart& s, Args&&... args);                   // since c++23

#include <memory>
#include <tuple>

#include "../types.h"

int main(int, char**) {
  // `std::out_ptr<>` requires `std::shared_ptr<>` with a deleter.
  {
    std::shared_ptr<int> sPtr;

    // expected-error-re@*:* {{static assertion failed due to requirement {{.*}}Using std::shared_ptr<> without a deleter in std::out_ptr is not supported.}}
    std::ignore = std::out_ptr(sPtr);
    // expected-error@*:* {{no matching conversion for functional-style cast from 'std::shared_ptr<int>' to 'std::out_ptr_t<shared_ptr<int>, _Ptr>' (aka 'out_ptr_t<std::shared_ptr<int>, int *>')}}
    std::ignore = std::out_ptr<int*>(sPtr);
  }

  return 0;
}

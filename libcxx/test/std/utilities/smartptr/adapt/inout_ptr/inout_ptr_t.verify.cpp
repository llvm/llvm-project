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

#include <memory>

int main(int, char**) {
  // `std::inout_ptr<>` does not support `std::shared_ptr<>`.
  {
    std::shared_ptr<int> sPtr;

    // expected-error-re@*:* {{static assertion failed due to requirement {{.*}}std::shared_ptr<> is not supported with std::inout_ptr.}}
    std::inout_ptr_t<std::shared_ptr<int>, int*>{sPtr};
  }

  return 0;
}

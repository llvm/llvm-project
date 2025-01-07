//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <memory>

// [out.ptr.t], class template out_ptr_t
// template<class Smart, class Pointer, class... Args>
//   class out_ptr_t;                                          // since c++23

#include <memory>

int main(int, char**) {
  // `std::out_ptr_t<>` requires `std::shared_ptr<>` with a deleter.
  {
    std::shared_ptr<int> sPtr;

    // expected-error-re@*:* {{static assertion failed due to requirement {{.*}}Using std::shared_ptr<> without a deleter in std::out_ptr is not supported.}}
    std::out_ptr_t<std::shared_ptr<int>, int*>{sPtr};
  }

  return 0;
}

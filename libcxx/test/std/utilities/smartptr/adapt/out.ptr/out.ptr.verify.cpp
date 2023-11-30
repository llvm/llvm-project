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

int main(int, char**) {
  // `std::out_ptr<>` requires `std::shared_ptr<>` with a deleter.
  {
    std::shared_ptr<int> sPtr;

    // expected-error-re@*:* {{{{(static_assert|static assertion)}} failed due to requirement '!__is_specialization_v<std::shared_ptr<int>, std::shared_ptr> || sizeof...(_Args) > 0'{{.*}}Specialization of std::shared_ptr<> requires a deleter.}}
    auto outSPtr1 = std::out_ptr(sPtr);
    // expected-error@*:* {{no matching conversion for functional-style cast from 'std::shared_ptr<int>' to 'std::out_ptr_t<shared_ptr<int>, _Ptr>' (aka 'out_ptr_t<std::shared_ptr<int>, int *>')}}
    auto outSPtr2 = std::out_ptr<int*>(sPtr);
  }

  return 0;
}

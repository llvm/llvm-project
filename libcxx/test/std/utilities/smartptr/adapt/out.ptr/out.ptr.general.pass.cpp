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
  {
    std::unique_ptr<int> uPtr;

    auto outUPtr1 = std::out_ptr(uPtr);
    (void)outUPtr1;
    auto outUPtr2 = std::out_ptr<int*>(uPtr);
    (void)outUPtr2;
  }
  {
    std::shared_ptr<int> sPtr;

    auto outSPtr1 = std::out_ptr(sPtr, [](auto* p) { delete p; });
    (void)outSPtr1;
    auto outSPtr2 = std::out_ptr<int*>(sPtr, [](auto* p) { delete p; });
    (void)outSPtr2;
  }

  return 0;
}

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

int main(int, char**) {
  {
    std::unique_ptr<int> uPtr;

    auto inoutUPtr1 = std::inout_ptr(uPtr);
    auto inoutUPtr2 = std::inout_ptr<int*>(uPtr);
  }
  {
    auto deleter = [](auto* p) { delete p; };
    std::unique_ptr<int, decltype(deleter)> uPtr;

    auto inoutUPtr1 = std::inout_ptr(uPtr, deleter);
    auto inoutUPtr2 = std::inout_ptr<int*>(uPtr, deleter);
  }

  return 0;
}

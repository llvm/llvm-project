//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20, c++23

// <functional>

// Test implementation-defined properties of std::not_fn<NTTP>.

#include <functional>
#include <type_traits>

struct NotEmptyFunctionObject {
  bool val = true;
  bool operator()() const; // not defined
};

void test() {
  using ResultWithEmptyFuncObject = decltype(std::not_fn<std::false_type{}>());
  static_assert(std::is_empty_v<ResultWithEmptyFuncObject>);

  using ResultWithNotEmptyFuncObject = decltype(std::not_fn<NotEmptyFunctionObject{}>());
  static_assert(std::is_empty_v<ResultWithNotEmptyFuncObject>);
}

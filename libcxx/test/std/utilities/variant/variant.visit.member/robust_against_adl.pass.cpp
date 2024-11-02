//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20, c++23
// The tested functionality needs deducing this.
// UNSUPPORTED: clang-16 || clang-17
// XFAIL: apple-clang

// <variant>

// class variant;
// template<class Self, class Visitor>
//   constexpr decltype(auto) visit(this Self&&, Visitor&&); // since C++26
// template<class R, class Self, class Visitor>
//   constexpr R visit(this Self&&, Visitor&&);              // since C++26

#include <variant>

#include "test_macros.h"

struct Incomplete;
template <class T>
struct Holder {
  T t;
};

constexpr bool test(bool do_it) {
  if (do_it) {
    std::variant<Holder<Incomplete>*, int> v = nullptr;

    v.visit([](auto) {});
    v.visit([](auto) -> Holder<Incomplete>* { return nullptr; });
    v.visit<void>([](auto) {});
    v.visit<void*>([](auto) -> Holder<Incomplete>* { return nullptr; });
  }
  return true;
}

int main(int, char**) {
  test(true);
  static_assert(test(true));

  return 0;
}

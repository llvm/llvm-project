//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <ranges>

// template<class F, class Tuple>
// constexpr void tuple-for-each(F&& f, Tuple&& t) { // exposition only

// LWG3755 tuple-for-each can call user-defined operator,

#include <ranges>
#include <tuple>
#include <cstdlib>

struct Evil {
  void operator,(Evil) { std::abort(); }
};

int main(int, char**) {
  std::tuple<int, int> t;
  std::ranges::__tuple_for_each([](int) { return Evil{}; }, t);

  return 0;
}

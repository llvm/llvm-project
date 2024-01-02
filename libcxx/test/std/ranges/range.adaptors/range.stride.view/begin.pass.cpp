//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// constexpr auto begin() requires(!__simple_view<_View>)
// constexpr auto begin() const requires range<const _View>

// Note: Checks here are augmented by checks in
// iterator/ctor.copy.pass.cpp.

#include <ranges>

#include "types.h"

template <class T>
concept HasConstBegin = requires(const T& ct) { ct.begin(); };

template <class T>
concept HasBegin = requires(T& t) { t.begin(); };

template <class T>
concept HasConstAndNonConstBegin = HasConstBegin<T> && requires(T& t, const T& ct) {
  requires !std::same_as<decltype(t.begin()), decltype(ct.begin())>;
};

template <class T>
concept HasOnlyNonConstBegin = HasBegin<T> && !HasConstBegin<T>;

template <class T>
concept HasOnlyConstBegin = HasConstBegin<T> && !HasConstAndNonConstBegin<T>;

struct NoConstView : std::ranges::view_base {
  int* begin();
  int* end();
};

struct UnsimpleConstView : std::ranges::view_base {
  double* begin();
  int* begin() const;

  double* end();
  int* end() const;
};
static_assert(HasOnlyNonConstBegin<std::ranges::stride_view<NoConstView>>);
static_assert(HasOnlyConstBegin<std::ranges::stride_view<BasicTestView<int*, int*>>>);
static_assert(HasConstAndNonConstBegin<std::ranges::stride_view<UnsimpleConstView>>);

int main(int, char**) {
  int buffer[] = {1, 2, 3};
  auto sv      = std::ranges::stride_view(BasicTestView<int*, int*>(buffer, buffer + 3), 1);
  assert(1 == *(sv.begin()));

  return 0;
}

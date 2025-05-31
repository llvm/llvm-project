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

#include <concepts>
#include <ranges>

#include "types.h"

template <class T>
concept HasConstBegin = requires(const T& ct) { ct.begin(); };

template <class T>
concept HasBegin = requires(T& t) { t.begin(); };

// _View has const-qualified begin and end methods and
// is _not_ a simple view.
template <class T>
concept HasConstAndNonConstBegin = requires(T& t, const T& ct) {
  // The return types for begin are different when this is const or not:
  // the iterator's _Const non-type-template parameter is true in the former
  // and false in the latter.
  requires !std::same_as<decltype(t.begin()), decltype(ct.begin())>;
};

// _View does not have const-qualified begin and end methods and
// is _not_ a simple view.
template <class T>
// There is a begin but it's not const qualified => Only non-const qualified begin.
concept HasOnlyNonConstBegin = HasBegin<T> && !HasConstBegin<T>;

// _View does have const-qualified begin and end methods and
// is a simple view.
template <class T>
// There is a const-qualified begin and there are not both const- and non-const qualified begin => Only const-qualified begin.
concept HasOnlyConstBegin = HasConstBegin<T> && !HasConstAndNonConstBegin<T>;

static_assert(HasOnlyNonConstBegin<std::ranges::stride_view<UnSimpleNoConstCommonView>>);
static_assert(HasOnlyConstBegin<std::ranges::stride_view<BasicTestView<int*, int*>>>);
static_assert(HasConstAndNonConstBegin<std::ranges::stride_view<UnSimpleConstView>>);

constexpr bool test() {
  const auto const_basic    = UnSimpleConstView();
  const auto sv_const_basic = std::ranges::stride_view(const_basic, 1);
  static_assert(std::same_as<decltype(*sv_const_basic.begin()), double&>);

  auto non_const_basic    = SimpleCommonConstView();
  auto sv_non_const_basic = std::ranges::stride_view(non_const_basic, 1);
  static_assert(std::same_as<decltype(*sv_non_const_basic.begin()), int&>);

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}
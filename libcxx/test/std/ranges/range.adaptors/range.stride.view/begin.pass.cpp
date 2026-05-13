//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// constexpr auto begin() requires(!simple-view<V>)
// constexpr auto begin() const requires range<const V>

// Note: Checks here are augmented by checks in
// iterator/ctor.copy.pass.cpp.

#include <cassert>
#include <concepts>
#include <ranges>

#include "types.h"

template <class T>
concept HasConstBegin = requires(const T& ct) { ct.begin(); };

template <class T>
concept HasBegin = requires(T& t) { t.begin(); };

template <class T>
concept HasConstAndNonConstBegin = requires(T& t, const T& ct) {
  // The return types for begin are different when this is const or not:
  // the iterator's _Const non-type-template parameter is true in the former
  // and false in the latter.
  requires !std::same_as<decltype(t.begin()), decltype(ct.begin())>;
};

template <class T>
// There is a begin but it's not const qualified => Only non-const qualified begin.
concept HasOnlyNonConstBegin = HasBegin<T> && !HasConstBegin<T>;

template <class T>
// There is a const-qualified begin and there are not both const- and non-const qualified begin => Only const-qualified begin.
concept HasOnlyConstBegin = HasConstBegin<T> && !HasConstAndNonConstBegin<T>;

static_assert(HasOnlyNonConstBegin<std::ranges::stride_view<UnSimpleNoConstCommonView>>);
static_assert(HasOnlyConstBegin<std::ranges::stride_view<SimpleCommonConstView>>);
static_assert(HasConstAndNonConstBegin<std::ranges::stride_view<UnSimpleConstView>>);

constexpr bool test() {
  {
    // Return type check for non-simple const view.
    const auto v  = UnSimpleConstView();
    const auto sv = std::ranges::stride_view(v, 1);
    static_assert(std::same_as<decltype(*sv.begin()), double&>);
  }
  {
    // Return type check for simple const view.
    auto v  = SimpleCommonConstView();
    auto sv = std::ranges::stride_view(v, 1);
    static_assert(std::same_as<decltype(*sv.begin()), int&>);
  }
  {
    // Verify begin() produces the first element with stride 1.
    int data[] = {10, 20, 30, 40, 50};
    auto v     = BasicTestView<int*, int*>{data, data + 5};
    auto sv    = std::ranges::stride_view(v, 1);
    assert(*sv.begin() == 10);
  }
  {
    // Verify begin() produces the first element with stride 3.
    int data[] = {10, 20, 30, 40, 50};
    auto v     = BasicTestView<int*, int*>{data, data + 5};
    auto sv    = std::ranges::stride_view(v, 3);
    assert(*sv.begin() == 10);
  }
  {
    // Verify iterating from begin with stride 2 produces correct elements.
    int data[] = {1, 2, 3, 4, 5};
    auto v     = BasicTestView<int*, int*>{data, data + 5};
    auto sv    = std::ranges::stride_view(v, 2);
    auto it    = sv.begin();
    assert(*it == 1);
    ++it;
    assert(*it == 3);
    ++it;
    assert(*it == 5);
    ++it;
    assert(it == sv.end());
  }
  {
    // Verify begin on forward range.
    int data[]    = {100, 200, 300};
    using FwdView = BasicTestView<forward_iterator<int*>, forward_iterator<int*>>;
    auto v        = FwdView{forward_iterator(data), forward_iterator(data + 3)};
    auto sv       = std::ranges::stride_view(v, 2);
    assert(*sv.begin() == 100);
    auto it = sv.begin();
    ++it;
    assert(*it == 300);
  }
  {
    // Empty range: begin() == end().
    int data[] = {1};
    auto v     = BasicTestView<int*, int*>{data, data};
    auto sv    = std::ranges::stride_view(v, 3);
    assert(sv.begin() == sv.end());
  }
  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}

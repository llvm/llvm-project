//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// constexpr auto end() requires (!simple-view<V> && !common_range<V>)
// constexpr auto end() requires (!simple-view<V> && common_range<V>)
// constexpr auto end() const requires range<const V>
// constexpr auto end() const requires common_range<const V>

#include <cassert>
#include <iterator>
#include <ranges>
#include <type_traits>
#include <utility>

#include "types.h"

// | simple | common |      v.end()     | as_const(v)
// |        |        |                  |   .end()
// |--------|--------|------------------|---------------
// |   Y    |   Y    |  iterator<true>  | iterator<true>
// |   Y    |   N    |  sentinel<true>  | sentinel<true>
// |   N    |   Y    |  iterator<false> | iterator<true>
// |   N    |   N    |  sentinel<false> | sentinel<true>

// !range<const V>
template <class T>
concept HasEnd = requires(T t) { t.end(); };

template <class T>
concept HasConstEnd = requires(const T ct) { ct.end(); };

struct NoConstEndView : TupleBufferView {
  using TupleBufferView::TupleBufferView;
  constexpr std::tuple<int>* begin() { return buffer_; }
  constexpr std::tuple<int>* end() { return buffer_ + size_; }
};

static_assert(HasEnd<std::ranges::elements_view<NoConstEndView, 0>>);
static_assert(!HasConstEnd<std::ranges::elements_view<NoConstEndView, 0>>);

constexpr bool test() {
  std::tuple<int> buffer[] = {{1}, {2}, {3}};

  // simple-view && common_view
  {
    SimpleCommon v{buffer};
    auto ev = std::views::elements<0>(v);

    auto it           = ev.begin();
    decltype(auto) st = ev.end();
    assert(st == it + 3);

    auto const_it           = std::as_const(ev).begin();
    decltype(auto) const_st = std::as_const(ev).end();
    assert(const_st == const_it + 3);

    // Both iterator<true>
    static_assert(std::same_as<decltype(st), decltype(const_st)>);
    static_assert(std::same_as<decltype(st), decltype(it)>);
    static_assert(std::same_as<decltype(const_st), decltype(const_it)>);
  }

  // simple-view && !common_view
  {
    SimpleNonCommon v{buffer};
    auto ev = std::views::elements<0>(v);

    auto it           = ev.begin();
    decltype(auto) st = ev.end();
    assert(st == it + 3);

    auto const_it           = std::as_const(ev).begin();
    decltype(auto) const_st = std::as_const(ev).end();
    assert(const_st == const_it + 3);

    // Both iterator<true>
    static_assert(std::same_as<decltype(st), decltype(const_st)>);
    static_assert(!std::same_as<decltype(st), decltype(it)>);
    static_assert(!std::same_as<decltype(const_st), decltype(const_it)>);
  }

  // !simple-view && common_view
  {
    NonSimpleCommon v{buffer};
    auto ev = std::views::elements<0>(v);

    auto it           = ev.begin();
    decltype(auto) st = ev.end();
    assert(st == it + 3);

    auto const_it           = std::as_const(ev).begin();
    decltype(auto) const_st = std::as_const(ev).end();
    assert(const_st == const_it + 3);

    // iterator<false> and iterator<true>
    static_assert(!std::same_as<decltype(st), decltype(const_st)>);
    static_assert(std::same_as<decltype(st), decltype(it)>);
    static_assert(std::same_as<decltype(const_st), decltype(const_it)>);
  }

  // !simple-view && !common_view
  {
    NonSimpleNonCommon v{buffer};
    auto ev = std::views::elements<0>(v);

    auto it           = ev.begin();
    decltype(auto) st = ev.end();
    assert(st == it + 3);

    auto const_it           = std::as_const(ev).begin();
    decltype(auto) const_st = std::as_const(ev).end();
    assert(const_st == const_it + 3);

    // sentinel<false> and sentinel<true>
    static_assert(!std::same_as<decltype(st), decltype(const_st)>);
    static_assert(!std::same_as<decltype(st), decltype(it)>);
    static_assert(!std::same_as<decltype(const_st), decltype(const_it)>);
  }

  // LWG 3406 elements_view::begin() and elements_view::end() have incompatible constraints
  {
    std::tuple<int, int> x[] = {{0, 0}};
    std::ranges::subrange r  = {std::counted_iterator(x, 1), std::default_sentinel};
    auto v                   = r | std::views::elements<0>;
    assert(v.begin() != v.end());
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}

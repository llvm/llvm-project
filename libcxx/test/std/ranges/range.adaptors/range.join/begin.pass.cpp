//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// constexpr auto begin();
// constexpr auto begin() const
//    requires forward_range<const V> &&
//             is_reference_v<range_reference_t<const V>> &&
//             input_range<range_reference_t<const V>>;

#include <algorithm>
#include <cassert>
#include <ranges>
#include <string_view>

#include "types.h"

struct NonSimpleParentView : std::ranges::view_base {
  ChildView* begin() { return nullptr; }
  const ChildView* begin() const;
  const ChildView* end() const;
};

struct SimpleParentView : std::ranges::view_base {
  const ChildView* begin() const;
  const ChildView* end() const;
};

struct ConstNotRange : std::ranges::view_base {
  const ChildView* begin();
  const ChildView* end();
};
static_assert(std::ranges::range<ConstNotRange>);
static_assert(!std::ranges::range<const ConstNotRange>);

template <class T>
concept HasConstBegin = requires(const T& t) { t.begin(); };

constexpr bool test() {
  int buffer[4][4] = {{1111, 2222, 3333, 4444}, {555, 666, 777, 888}, {99, 1010, 1111, 1212}, {13, 14, 15, 16}};

  {
    ChildView children[4] = {ChildView(buffer[0]), ChildView(buffer[1]), ChildView(buffer[2]), ChildView(buffer[3])};
    auto jv = std::ranges::join_view(ParentView{children});
    assert(*jv.begin() == 1111);
  }

  {
    CopyableChild children[4] = {CopyableChild(buffer[0], 4), CopyableChild(buffer[1], 0), CopyableChild(buffer[2], 1),
                                 CopyableChild(buffer[3], 0)};
    auto jv = std::ranges::join_view(ParentView{children});
    assert(*jv.begin() == 1111);
  }

  // Parent is empty.
  {
    CopyableChild children[4] = {CopyableChild(buffer[0]), CopyableChild(buffer[1]), CopyableChild(buffer[2]),
                                 CopyableChild(buffer[3])};
    std::ranges::join_view jv(ParentView(children, 0));
    assert(jv.begin() == jv.end());
  }

  // Parent size is one.
  {
    CopyableChild children[1] = {CopyableChild(buffer[0])};
    std::ranges::join_view jv(ParentView(children, 1));
    assert(*jv.begin() == 1111);
  }

  // Parent and child size is one.
  {
    CopyableChild children[1] = {CopyableChild(buffer[0], 1)};
    std::ranges::join_view jv(ParentView(children, 1));
    assert(*jv.begin() == 1111);
  }

  // Parent size is one child is empty
  {
    CopyableChild children[1] = {CopyableChild(buffer[0], 0)};
    std::ranges::join_view jv(ParentView(children, 1));
    assert(jv.begin() == jv.end());
  }

  // Has all empty children.
  {
    CopyableChild children[4] = {CopyableChild(buffer[0], 0), CopyableChild(buffer[1], 0), CopyableChild(buffer[2], 0),
                                 CopyableChild(buffer[3], 0)};
    auto jv = std::ranges::join_view(ParentView{children});
    assert(jv.begin() == jv.end());
  }

  // First child is empty, others are not.
  {
    CopyableChild children[4] = {CopyableChild(buffer[0], 4), CopyableChild(buffer[1], 0), CopyableChild(buffer[2], 0),
                                 CopyableChild(buffer[3], 0)};
    auto jv = std::ranges::join_view(ParentView{children});
    assert(*jv.begin() == 1111);
  }

  // Last child is empty, others are not.
  {
    CopyableChild children[4] = {CopyableChild(buffer[0], 4), CopyableChild(buffer[1], 4), CopyableChild(buffer[2], 4),
                                 CopyableChild(buffer[3], 0)};
    auto jv = std::ranges::join_view(ParentView{children});
    assert(*jv.begin() == 1111);
  }

  {
    std::ranges::join_view jv(buffer);
    assert(*jv.begin() == 1111);
  }

  {
    const std::ranges::join_view jv(buffer);
    assert(*jv.begin() == 1111);
    static_assert(HasConstBegin<decltype(jv)>);
  }

  // !forward_range<const V>
  {
    std::ranges::join_view jv{ConstNotRange{}};
    static_assert(!HasConstBegin<decltype(jv)>);
  }

  // !is_reference_v<range_reference_t<const V>>
  {
    auto innerRValueRange = std::views::iota(0, 5) | std::views::transform([](int) { return ChildView{}; });
    static_assert(!std::is_reference_v<std::ranges::range_reference_t<const decltype(innerRValueRange)>>);
    std::ranges::join_view jv{innerRValueRange};
    static_assert(!HasConstBegin<decltype(jv)>);
  }

  // !simple-view<V>
  {
    std::ranges::join_view<NonSimpleParentView> jv;
    static_assert(!std::same_as<decltype(jv.begin()), decltype(std::as_const(jv).begin())>);
  }

  // simple-view<V> && is_reference_v<range_reference_t<V>>;
  {
    std::ranges::join_view<SimpleParentView> jv;
    static_assert(std::same_as<decltype(jv.begin()), decltype(std::as_const(jv).begin())>);
  }

  // Check stashing iterators (LWG3698: regex_iterator and join_view don't work together very well)
  {
    std::ranges::join_view<StashingRange> jv;
    assert(std::ranges::equal(std::views::counted(jv.begin(), 10), std::string_view{"aababcabcd"}));
  }

  // LWG3700: The `const begin` of the `join_view` family does not require `InnerRng` to be a range
  {
    std::ranges::join_view<ConstNonJoinableRange> jv;
    static_assert(!HasConstBegin<decltype(jv)>);
  }

  // Check example from LWG3700
  {
    auto r = std::views::iota(0, 5) | std::views::split(1);
    auto s = std::views::single(r);
    auto j = s | std::views::join;
    auto f = j.front();
    assert(std::ranges::equal(f, std::views::single(0)));
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}

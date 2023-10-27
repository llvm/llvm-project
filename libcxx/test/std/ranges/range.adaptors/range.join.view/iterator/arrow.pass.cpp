//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: !c++experimental

// constexpr InnerIter operator->() const
//   requires has-arrow<InnerIter> && copyable<InnerIter>;

#include <cassert>
#include <ranges>

#include "../types.h"

template <class T>
concept HasArrow = std::input_iterator<T> && (std::is_pointer_v<T> || requires(T i) { i.operator->(); });

template <class Base>
struct move_only_input_iter_with_arrow {
  Base it_;

  using value_type = std::iter_value_t<Base>;
  using difference_type = std::intptr_t;
  using iterator_concept = std::input_iterator_tag;

  constexpr move_only_input_iter_with_arrow(Base it) : it_(std::move(it)) {}
  constexpr move_only_input_iter_with_arrow(move_only_input_iter_with_arrow&&) = default;
  constexpr move_only_input_iter_with_arrow(const move_only_input_iter_with_arrow&) = delete;
  constexpr move_only_input_iter_with_arrow& operator=(move_only_input_iter_with_arrow&&) = default;
  constexpr move_only_input_iter_with_arrow& operator=(const move_only_input_iter_with_arrow&) = delete;

  constexpr move_only_input_iter_with_arrow& operator++() {
    ++it_;
    return *this;
  }
  constexpr void operator++(int) { ++it_; }

  constexpr std::iter_reference_t<Base> operator*() const { return *it_; }
  constexpr auto operator->() const
    requires(HasArrow<Base> && std::copyable<Base>) {
    return it_;
  }
};
static_assert(!std::copyable<move_only_input_iter_with_arrow<int*>>);
static_assert(std::input_iterator<move_only_input_iter_with_arrow<int*>>);

template <class Base>
struct move_iter_sentinel {
  Base it_;
  explicit move_iter_sentinel() = default;
  constexpr move_iter_sentinel(Base it) : it_(std::move(it)) {}
  constexpr bool operator==(const move_only_input_iter_with_arrow<Base>& other) const { return it_ == other.it_; }
};
static_assert(std::sentinel_for<move_iter_sentinel<int*>, move_only_input_iter_with_arrow<int*>>);

struct MoveOnlyIterInner : BufferView<move_only_input_iter_with_arrow<Box*>, move_iter_sentinel<Box*>> {
  using BufferView::BufferView;

  using iterator = move_only_input_iter_with_arrow<Box*>;
  using sentinel = move_iter_sentinel<Box*>;

  iterator begin() const { return data_; }
  sentinel end() const { return sentinel{data_ + size_}; }
};
static_assert(std::ranges::input_range<MoveOnlyIterInner>);

template <class Base>
struct arrow_input_iter {
  Base it_;

  using value_type = std::iter_value_t<Base>;
  using difference_type = std::intptr_t;
  using iterator_concept = std::input_iterator_tag;

  arrow_input_iter() = default;
  constexpr arrow_input_iter(Base it) : it_(std::move(it)) {}

  constexpr arrow_input_iter& operator++() {
    ++it_;
    return *this;
  }
  constexpr void operator++(int) { ++it_; }

  constexpr std::iter_reference_t<Base> operator*() const { return *it_; }
  constexpr auto operator->() const { return it_; }

  friend constexpr bool operator==(const arrow_input_iter& x, const arrow_input_iter& y) = default;
};

using ArrowInner = BufferView<arrow_input_iter<Box*>>;
static_assert(std::ranges::input_range<ArrowInner>);
static_assert(HasArrow<std::ranges::iterator_t<ArrowInner>>);

constexpr bool test() {
  Box buffer[4][4] = {{{1111}, {2222}, {3333}, {4444}},
                      {{555}, {666}, {777}, {888}},
                      {{99}, {1010}, {1111}, {1212}},
                      {{13}, {14}, {15}, {16}}};

  {
    // Copyable input iterator with arrow.
    using BoxView = ValueView<Box>;
    ValueView<Box> children[4] = {BoxView(buffer[0]), BoxView(buffer[1]), BoxView(buffer[2]), BoxView(buffer[3])};
    std::ranges::join_view jv(ValueView<ValueView<Box>>{children});
    assert(jv.begin()->x == 1111);
    static_assert(HasArrow<decltype(jv.begin())>);
  }

  {
    std::ranges::join_view jv(buffer);
    assert(jv.begin()->x == 1111);
    static_assert(HasArrow<decltype(jv.begin())>);
  }

  {
    const std::ranges::join_view jv(buffer);
    assert(jv.begin()->x == 1111);
    static_assert(HasArrow<decltype(jv.begin())>);
  }

  {
    // LWG3500 `join_view::iterator::operator->()` is bogus
    // `operator->` should not be defined if inner iterator is not copyable
    // has-arrow<InnerIter> && !copyable<InnerIter>
    static_assert(HasArrow<move_only_input_iter_with_arrow<int*>>);
    MoveOnlyIterInner inners[2] = {buffer[0], buffer[1]};
    std::ranges::join_view jv{inners};
    static_assert(HasArrow<decltype(std::ranges::begin(inners[0]))>);
    static_assert(!HasArrow<decltype(jv.begin())>);
  }

  {
    // LWG3500 `join_view::iterator::operator->()` is bogus
    // `operator->` should not be defined if inner iterator does not have `operator->`
    // !has-arrow<InnerIter> && copyable<InnerIter>
    using Inner = BufferView<forward_iterator<Box*>>;
    Inner inners[2] = {buffer[0], buffer[1]};
    std::ranges::join_view jv{inners};
    static_assert(!HasArrow<decltype(std::ranges::begin(inners[0]))>);
    static_assert(!HasArrow<decltype(jv.begin())>);
  }

  {
    // arrow returns inner iterator
    ArrowInner inners[2] = {buffer[0], buffer[1]};
    std::ranges::join_view jv{inners};
    static_assert(HasArrow<decltype(std::ranges::begin(inners[0]))>);
    static_assert(HasArrow<decltype(jv.begin())>);

    auto jv_it = jv.begin();
    std::same_as<arrow_input_iter<Box*>> auto arrow_it = jv_it.operator->();
    assert(arrow_it->x == 1111);
  }
  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}

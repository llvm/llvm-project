//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// constexpr sentinel(sentinel<!Const> s);
//             requires Const && convertible_to<sentinel_t<V>, sentinel_t<Base>>;

#include <cassert>
#include <ranges>

#include "../types.h"
#include "test_range.h"

template <class T>
struct convertible_sentinel_wrapper {
  explicit convertible_sentinel_wrapper() = default;
  constexpr convertible_sentinel_wrapper(const T& it) : it_(it) {}

  template <class U>
    requires std::convertible_to<const U&, T>
  constexpr convertible_sentinel_wrapper(const convertible_sentinel_wrapper<U>& other) : it_(other.it_) {}

  constexpr friend bool operator==(convertible_sentinel_wrapper const& self, const T& other) {
    return self.it_ == other;
  }
  T it_;
};

struct ConstConvertibleView : BufferView<BufferView<int*>*> {
  using BufferView<BufferView<int*>*>::BufferView;

  using sentinel = convertible_sentinel_wrapper<BufferView<int*>*>;
  using const_sentinel = convertible_sentinel_wrapper<const BufferView<int*>*>;

  constexpr BufferView<int*>* begin() { return data_; }
  constexpr const BufferView<int*>* begin() const { return data_; }
  constexpr sentinel end() { return sentinel(data_ + size_); }
  constexpr const_sentinel end() const { return const_sentinel(data_ + size_); }
};
static_assert(!std::ranges::common_range<ConstConvertibleView>);
static_assert(std::convertible_to<std::ranges::sentinel_t<ConstConvertibleView>,
                                  std::ranges::sentinel_t<ConstConvertibleView const>>);
static_assert(!simple_view<ConstConvertibleView>);

constexpr bool test() {
  int buffer[4][4] = {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}};
  {
    BufferView<int*> inners[] = {buffer[0], buffer[1], buffer[2]};
    ConstConvertibleView outer(inners);
    std::ranges::join_view jv(outer);
    auto sent1 = jv.end();
    std::ranges::sentinel_t<const decltype(jv)> sent2 = sent1;
    assert(std::as_const(jv).begin() != sent2);
    assert(std::ranges::next(std::as_const(jv).begin(), 12) == sent2);

    // We cannot create a non-const sentinel from a const sentinel.
    static_assert(!std::constructible_from<decltype(sent1), decltype(sent2)>);
  }

  {
    // cannot create a const sentinel from a non-const sentinel if the underlying
    // const sentinel cannot be created from the underlying non-const sentinel
    using Inner = BufferView<int*>;
    using ConstInconvertibleOuter =
        BufferView<forward_iterator<const Inner*>, sentinel_wrapper<forward_iterator<const Inner*>>,
                   bidirectional_iterator<Inner*>, sentinel_wrapper<bidirectional_iterator<Inner*>>>;
    using JoinView = std::ranges::join_view<ConstInconvertibleOuter>;
    using sentinel_t     = std::ranges::sentinel_t<JoinView>;
    using const_sentinel_t = std::ranges::sentinel_t<const JoinView>;
    static_assert(!std::constructible_from<sentinel_t, const_sentinel_t>);
    static_assert(!std::constructible_from<const_sentinel_t, sentinel_t>);
  }
  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}

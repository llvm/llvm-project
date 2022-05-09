//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20
// UNSUPPORTED: libcpp-has-no-incomplete-ranges

// constexpr iterator& operator++();
// constexpr void operator++(int);
// constexpr iterator operator++(int) requires all_forward<Const, Views...>;

#include <array>
#include <cassert>
#include <ranges>
#include <tuple>

#include "../types.h"

struct InputRange : IntBufferView {
  using IntBufferView::IntBufferView;
  using iterator = cpp20_input_iterator<int*>;
  constexpr iterator begin() const { return iterator(buffer_); }
  constexpr sentinel_wrapper<iterator> end() const { return sentinel_wrapper<iterator>(iterator(buffer_ + size_)); }
};

constexpr bool test() {
  std::array a{1, 2, 3, 4};
  std::array b{4.1, 3.2, 4.3};
  {
    // random/contiguous
    std::ranges::zip_view v(a, b, std::views::iota(0, 5));
    auto it = v.begin();
    using Iter = decltype(it);

    assert(&(std::get<0>(*it)) == &(a[0]));
    assert(&(std::get<1>(*it)) == &(b[0]));
    assert(std::get<2>(*it) == 0);

    static_assert(std::is_same_v<decltype(++it), Iter&>);

    auto& it_ref = ++it;
    assert(&it_ref == &it);

    assert(&(std::get<0>(*it)) == &(a[1]));
    assert(&(std::get<1>(*it)) == &(b[1]));
    assert(std::get<2>(*it) == 1);

    static_assert(std::is_same_v<decltype(it++), Iter>);
    auto original = it;
    auto copy = it++;
    assert(original == copy);
    assert(&(std::get<0>(*it)) == &(a[2]));
    assert(&(std::get<1>(*it)) == &(b[2]));
    assert(std::get<2>(*it) == 2);
  }

  {
    //  bidi
    int buffer[2] = {1, 2};

    std::ranges::zip_view v(BidiCommonView{buffer});
    auto it = v.begin();
    using Iter = decltype(it);

    assert(&(std::get<0>(*it)) == &(buffer[0]));

    static_assert(std::is_same_v<decltype(++it), Iter&>);
    auto& it_ref = ++it;
    assert(&it_ref == &it);
    assert(&(std::get<0>(*it)) == &(buffer[1]));

    static_assert(std::is_same_v<decltype(it++), Iter>);
    auto original = it;
    auto copy = it++;
    assert(copy == original);
    assert(&(std::get<0>(*it)) == &(buffer[2]));
  }

  {
    //  forward
    int buffer[2] = {1, 2};

    std::ranges::zip_view v(ForwardSizedView{buffer});
    auto it = v.begin();
    using Iter = decltype(it);

    assert(&(std::get<0>(*it)) == &(buffer[0]));

    static_assert(std::is_same_v<decltype(++it), Iter&>);
    auto& it_ref = ++it;
    assert(&it_ref == &it);
    assert(&(std::get<0>(*it)) == &(buffer[1]));

    static_assert(std::is_same_v<decltype(it++), Iter>);
    auto original = it;
    auto copy = it++;
    assert(copy == original);
    assert(&(std::get<0>(*it)) == &(buffer[2]));
  }

  {
    // all input+
    int buffer[3] = {4, 5, 6};
    std::ranges::zip_view v(a, InputRange{buffer});
    auto it = v.begin();
    using Iter = decltype(it);

    assert(&(std::get<0>(*it)) == &(a[0]));
    assert(&(std::get<1>(*it)) == &(buffer[0]));

    static_assert(std::is_same_v<decltype(++it), Iter&>);
    auto& it_ref = ++it;
    assert(&it_ref == &it);
    assert(&(std::get<0>(*it)) == &(a[1]));
    assert(&(std::get<1>(*it)) == &(buffer[1]));

    static_assert(std::is_same_v<decltype(it++), void>);
    it++;
    assert(&(std::get<0>(*it)) == &(a[2]));
    assert(&(std::get<1>(*it)) == &(buffer[2]));
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}

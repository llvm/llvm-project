//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// constexpr iterator& operator++();
// constexpr void operator++(int);
// constexpr iterator operator++(int) requires forward_range<Base>;;

#include <array>
#include <cassert>
#include <ranges>

#include "../types.h"

struct InputRange : IntBufferView {
  using IntBufferView::IntBufferView;
  using iterator = cpp20_input_iterator<int*>;
  constexpr iterator begin() const { return iterator(buffer_); }
  constexpr sentinel_wrapper<iterator> end() const { return sentinel_wrapper<iterator>(iterator(buffer_ + size_)); }
};

template <class View>
constexpr void testForwardPlus() {
  int buffer[] = {1, 2, 3, 4};

  std::ranges::zip_transform_view v(GetFirst{}, View{buffer}, View{buffer});
  auto it    = v.begin();
  using Iter = decltype(it);

  assert(&(*it) == &(buffer[0]));

  std::same_as<Iter&> decltype(auto) it_ref = ++it;
  assert(&it_ref == &it);
  assert(&(*it) == &(buffer[1]));

  static_assert(std::is_same_v<decltype(it++), Iter>);
  auto original                          = it;
  std::same_as<Iter> decltype(auto) copy = it++;
  assert(original == copy);
  assert(&(*it) == &(buffer[2]));
}

constexpr bool test() {
  testForwardPlus<SizedRandomAccessView>();
  testForwardPlus<BidiCommonView>();
  testForwardPlus<ForwardSizedView>();

  {
    // test input_range
    int buffer[3] = {4, 5, 6};
    std::ranges::zip_transform_view v(MakeTuple{}, InputRange{buffer}, InputRange{buffer});
    auto it    = v.begin();
    using Iter = decltype(it);

    assert(*it == std::tuple(4, 4));

    std::same_as<Iter&> decltype(auto) it_ref = ++it;
    assert(&it_ref == &it);
    assert(*it == std::tuple(5, 5));

    static_assert(std::is_same_v<decltype(it++), void>);
    it++;
    assert(*it == std::tuple(6, 6));
  }

  int buffer[] = {1, 2, 3, 4, 5, 6};

  {
    // one range
    std::ranges::zip_transform_view v(MakeTuple{}, SimpleCommon{buffer});
    auto it    = v.begin();
    using Iter = decltype(it);

    std::same_as<Iter&> decltype(auto) it_ref = ++it;
    assert(&it_ref == &it);

    assert(*it == std::tuple(2));

    auto original                         = it;
    std::same_as<Iter> decltype(auto) it2 = it++;
    assert(original == it2);
    assert(*it == std::tuple(3));
  }

  {
    // two ranges
    std::ranges::zip_transform_view v(GetFirst{}, SimpleCommon{buffer}, std::views::iota(0));
    auto it    = v.begin();
    using Iter = decltype(it);

    std::same_as<Iter&> decltype(auto) it_ref = ++it;
    assert(&it_ref == &it);

    assert(*it == 2);

    auto original                         = it;
    std::same_as<Iter> decltype(auto) it2 = it++;
    assert(original == it2);
    assert(*it == 3);
  }

  {
    // three ranges
    std::ranges::zip_transform_view v(Tie{}, SimpleCommon{buffer}, SimpleCommon{buffer}, std::ranges::repeat_view(2.));
    auto it    = v.begin();
    using Iter = decltype(it);

    std::same_as<Iter&> decltype(auto) it_ref = ++it;
    assert(&it_ref == &it);

    assert(*it == std::tuple(2, 2, 2.0));

    auto original                         = it;
    std::same_as<Iter> decltype(auto) it2 = it++;
    assert(original == it2);
    assert(*it == std::tuple(3, 3, 2.0));
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}

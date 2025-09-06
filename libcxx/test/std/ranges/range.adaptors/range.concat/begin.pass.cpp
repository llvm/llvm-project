//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

#include <ranges>
#include <vector>

#include <cassert>
#include "test_iterators.h"
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

constexpr void tests() {
  // check the case of simple view
  {
    int buffer[4] = {1, 2, 3, 4};
    std::ranges::concat_view v(SimpleCommon{buffer}, SimpleCommon{buffer});
    static_assert(std::is_same_v<decltype(v.begin()), decltype(std::as_const(v).begin())>);
    assert(v.begin() == std::as_const(v).begin());

    using View = decltype(v);
    static_assert(HasOnlyConstBegin<View>);
    static_assert(!HasOnlyNonConstBegin<View>);
    static_assert(!HasConstAndNonConstBegin<View>);
  }

  // not all underlying ranges model simple view
  {
    int buffer[4] = {1, 2, 3, 4};
    std::ranges::concat_view v(SimpleCommon{buffer}, NonSimpleNonCommon{buffer});
    static_assert(!std::is_same_v<decltype(v.begin()), decltype(std::as_const(v).begin())>);
    assert(v.begin() == std::as_const(v).begin());

    using View = decltype(v);
    static_assert(!HasOnlyConstBegin<View>);
    static_assert(!HasOnlyNonConstBegin<View>);
    static_assert(HasConstAndNonConstBegin<View>);
  }

  // first view is empty
  {
    std::vector<int> v1;
    std::vector<int> v2 = {1, 2, 3, 4};
    std::ranges::concat_view view(v1, v2);
    auto it = view.begin();
    assert(*it == 1);
    assert(it + 4 == view.end());
  }

  // first few views is empty
  {
    std::vector<int> v1;
    std::vector<int> v2;
    std::vector<int> v3 = {1, 2, 3, 4};
    std::ranges::concat_view view(v1, v2, v3);
    auto it = view.begin();
    assert(*it == 1);
    assert(it + 4 == view.end());
  }
}

constexpr bool test() {
  tests();
  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}

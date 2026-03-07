//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// constexpr __iterator(__iterator<!_Const> __i)
//     requires _Const && (convertible_to<iterator_t<_Views>, iterator_t<const _Views>> && ...)

#include <cassert>
#include <ranges>

#include "../../range_adaptor_types.h"

using ConstIterIncompatibleView =
    BasicView<forward_iterator<int*>,
              forward_iterator<int*>,
              random_access_iterator<const int*>,
              random_access_iterator<const int*>>;
static_assert(!std::convertible_to<std::ranges::iterator_t<ConstIterIncompatibleView>,
                                   std::ranges::iterator_t<const ConstIterIncompatibleView>>);

constexpr bool test() {
  int buffer_1[3] = {1, 2, 3};
  int buffer_2[3] = {4, 5, 6};

  {
    std::ranges::concat_view v(NonSimpleCommon{buffer_1}, NonSimpleCommonRandomAccessSized{buffer_2});
    auto iter1                                       = v.begin();
    iter1++;
    std::ranges::iterator_t<const decltype(v)> iter2 = iter1;
    assert(iter1 == iter2);
    assert(*iter1 == 2);
    assert(*iter2 == 2);

    static_assert(!std::is_same_v<decltype(iter1), decltype(iter2)>);

    // We cannot create a non-const iterator from a const iterator.
    static_assert(!std::constructible_from<decltype(iter1), decltype(iter2)>);
  }

  {
    // iter1 in the second range
    std::ranges::concat_view v(NonSimpleCommon{buffer_1}, NonSimpleCommonRandomAccessSized{buffer_2});
    auto iter1 = v.begin();
    iter1++;
    iter1++;
    iter1++;
    std::ranges::iterator_t<const decltype(v)> iter2 = iter1;
    assert(iter1 == iter2);
    assert(*iter1 == 4);
    assert(*iter2 == 4);

    static_assert(!std::is_same_v<decltype(iter1), decltype(iter2)>);

    // We cannot create a non-const iterator from a const iterator.
    static_assert(!std::constructible_from<decltype(iter1), decltype(iter2)>);
  }

  {
    // underlying non-const to const not convertible
    std::ranges::concat_view v(ConstIterIncompatibleView{buffer_1}, NonSimpleCommon{buffer_2});
    auto iter1 = v.begin();
    auto iter2 = std::as_const(v).begin();

    static_assert(!std::is_same_v<decltype(iter1), decltype(iter2)>);
    static_assert(!std::constructible_from<decltype(iter1), decltype(iter2)>);
    static_assert(!std::constructible_from<decltype(iter2), decltype(iter1)>);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: !c++experimental

// constexpr iterator(iterator<!Const> i)
//             requires Const &&
//                      convertible_to<iterator_t<V>, OuterIter> &&
//                      convertible_to<iterator_t<InnerRng>, InnerIter>;

#include <cassert>
#include <ranges>

#include "test_macros.h"
#include "../types.h"

using ConstCompatibleInner = BufferView<int*>;

using ConstIncompatibleInner = BufferView<forward_iterator<const int*>, forward_iterator<const int*>,
                                          bidirectional_iterator<int*>, bidirectional_iterator<int*>>;

template <class Inner>
using ConstCompatibleOuter = BufferView<const Inner*, const Inner*, Inner*, Inner*>;

template <class Inner>
using ConstIncompatibleOuter = BufferView<forward_iterator<const Inner*>, forward_iterator<const Inner*>,
                                          bidirectional_iterator<Inner*>, bidirectional_iterator<Inner*>>;

constexpr bool test() {
  int buffer[4][4] = {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}};
  {
    CopyableChild children[4] = {CopyableChild(buffer[0]), CopyableChild(buffer[1]), CopyableChild(buffer[2]),
                                 CopyableChild(buffer[3])};
    std::ranges::join_view jv(CopyableParent{children});
    auto iter1 = jv.begin();
    using iterator = decltype(iter1);
    using const_iterator = decltype(std::as_const(jv).begin());
    static_assert(!std::is_same_v<iterator, const_iterator>);
    const_iterator iter2 = iter1;
    assert(iter1 == iter2);

    // We cannot create a non-const iterator from a const iterator.
    static_assert(!std::constructible_from<iterator, const_iterator>);
  }

  // !convertible_to<inner_iterator, inner_const_iterator>>;
  {
    ConstIncompatibleInner inners[2] = {buffer[0], buffer[1]};
    ConstCompatibleOuter<ConstIncompatibleInner> outer{inners};
    std::ranges::join_view jv(outer);
    using iterator = decltype(jv.begin());
    using const_iterator = decltype(std::as_const(jv).begin());
    static_assert(!std::is_same_v<iterator, const_iterator>);

    static_assert(!std::constructible_from<const_iterator, iterator>);
    static_assert(!std::constructible_from<iterator, const_iterator>);
  }

  // !convertible_to<outer_iterator, outer_const_iterator>>;
  {
    ConstCompatibleInner inners[2] = {buffer[0], buffer[1]};
    ConstIncompatibleOuter<ConstCompatibleInner> outer{inners};
    std::ranges::join_view jv(outer);
    using iterator = decltype(jv.begin());
    using const_iterator = decltype(std::as_const(jv).begin());
    static_assert(!std::is_same_v<iterator, const_iterator>);

    static_assert(!std::constructible_from<const_iterator, iterator>);
    static_assert(!std::constructible_from<iterator, const_iterator>);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}

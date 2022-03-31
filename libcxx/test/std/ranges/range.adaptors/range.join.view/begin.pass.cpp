//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-has-no-incomplete-ranges

// constexpr auto begin();
// constexpr auto begin() const;

#include <cassert>
#include <ranges>

#include "test_macros.h"
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

constexpr bool test() {
  int buffer[4][4] = {{1111, 2222, 3333, 4444}, {555, 666, 777, 888}, {99, 1010, 1111, 1212}, {13, 14, 15, 16}};

  {
    ChildView children[4] = {ChildView(buffer[0]), ChildView(buffer[1]), ChildView(buffer[2]), ChildView(buffer[3])};
    auto jv = std::ranges::join_view(ParentView{children});
    assert(*jv.begin() == 1111);
  }

  {
    CopyableChild children[4] = {CopyableChild(buffer[0], 4), CopyableChild(buffer[1], 0), CopyableChild(buffer[2], 1), CopyableChild(buffer[3], 0)};
    auto jv = std::ranges::join_view(ParentView{children});
    assert(*jv.begin() == 1111);
  }
  // Parent is empty.
  {
    CopyableChild children[4] = {CopyableChild(buffer[0]), CopyableChild(buffer[1]), CopyableChild(buffer[2]), CopyableChild(buffer[3])};
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
    CopyableChild children[4] = {CopyableChild(buffer[0], 0), CopyableChild(buffer[1], 0), CopyableChild(buffer[2], 0), CopyableChild(buffer[3], 0)};
    auto jv = std::ranges::join_view(ParentView{children});
    assert(jv.begin() == jv.end());
  }
  // First child is empty, others are not.
  {
    CopyableChild children[4] = {CopyableChild(buffer[0], 4), CopyableChild(buffer[1], 0), CopyableChild(buffer[2], 0), CopyableChild(buffer[3], 0)};
    auto jv = std::ranges::join_view(ParentView{children});
    assert(*jv.begin() == 1111);
  }
  // Last child is empty, others are not.
  {
    CopyableChild children[4] = {CopyableChild(buffer[0], 4), CopyableChild(buffer[1], 4), CopyableChild(buffer[2], 4), CopyableChild(buffer[3], 0)};
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
  }

  // !simple-view<V>
  {
    std::ranges::join_view<NonSimpleParentView> jv;
    static_assert(!std::same_as<decltype(jv.begin()),
                                decltype(std::as_const(jv).begin())>);
  }

  // simple-view<V> && is_reference_v<range_reference_t<V>>;
  {
    std::ranges::join_view<SimpleParentView> jv;
    static_assert(std::same_as<decltype(jv.begin()),
                               decltype(std::as_const(jv).begin())>);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}

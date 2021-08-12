//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts
// UNSUPPORTED: libcpp-has-no-incomplete-ranges

// common_view() requires default_initializable<V> = default;

#include <ranges>
#include <cassert>

#include "test_iterators.h"
#include "test_range.h"

int globalBuffer[4] = {1,2,3,4};

struct ContiguousView : std::ranges::view_base {
  int *ptr_;
  constexpr ContiguousView(int* ptr) : ptr_(ptr) {}
  constexpr ContiguousView(ContiguousView&&) = default;
  constexpr ContiguousView& operator=(ContiguousView&&) = default;
  friend constexpr int* begin(ContiguousView& view) { return view.ptr_; }
  friend constexpr int* begin(ContiguousView const& view) { return view.ptr_; }
  friend constexpr sentinel_wrapper<int*> end(ContiguousView& view) {
    return sentinel_wrapper<int*>{view.ptr_ + 8};
  }
  friend constexpr sentinel_wrapper<int*> end(ContiguousView const& view) {
    return sentinel_wrapper<int*>{view.ptr_ + 8};
  }
};

struct CopyableView : std::ranges::view_base {
  int *ptr_;
  constexpr CopyableView(int* ptr = globalBuffer) : ptr_(ptr) {}
  friend constexpr int* begin(CopyableView& view) { return view.ptr_; }
  friend constexpr int* begin(CopyableView const& view) { return view.ptr_; }
  friend constexpr sentinel_wrapper<int*> end(CopyableView& view) {
    return sentinel_wrapper<int*>{view.ptr_ + 4};
  }
  friend constexpr sentinel_wrapper<int*> end(CopyableView const& view) {
    return sentinel_wrapper<int*>{view.ptr_ + 4};
  }
};

struct DefaultConstructibleView : std::ranges::view_base {
  DefaultConstructibleView();
  friend int* begin(DefaultConstructibleView& view);
  friend int* begin(DefaultConstructibleView const& view);
  friend sentinel_wrapper<int*> end(DefaultConstructibleView& view);
  friend sentinel_wrapper<int*> end(DefaultConstructibleView const& view);
};

int main(int, char**) {
  static_assert(!std::default_initializable<std::ranges::common_view<ContiguousView>>);
  static_assert( std::default_initializable<std::ranges::common_view<DefaultConstructibleView>>);

  std::ranges::common_view<CopyableView> common;
  assert(*common.begin() == 1);

  return 0;
}

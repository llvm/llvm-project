//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts
// UNSUPPORTED: gcc-10
// UNSUPPORTED: libcpp-has-no-incomplete-ranges

// constexpr auto size() requires sized_range<V>
// constexpr auto size() const requires sized_range<const V>

#include <ranges>
#include <cassert>

#include "test_iterators.h"
#include "test_range.h"

struct CopyableView : std::ranges::view_base {
  int *ptr_;
  constexpr CopyableView(int* ptr) : ptr_(ptr) {}
  friend constexpr int* begin(CopyableView& view) { return view.ptr_; }
  friend constexpr int* begin(CopyableView const& view) { return view.ptr_; }
  friend constexpr sentinel_wrapper<int*> end(CopyableView& view) {
    return sentinel_wrapper<int*>{view.ptr_ + 8};
  }
  friend constexpr sentinel_wrapper<int*> end(CopyableView const& view) {
    return sentinel_wrapper<int*>{view.ptr_ + 8};
  }
};

using ForwardIter = forward_iterator<int*>;
struct SizedForwardView : std::ranges::view_base {
  int *ptr_;
  constexpr SizedForwardView(int* ptr) : ptr_(ptr) {}
  friend constexpr ForwardIter begin(SizedForwardView& view) { return ForwardIter(view.ptr_); }
  friend constexpr ForwardIter begin(SizedForwardView const& view) { return ForwardIter(view.ptr_); }
  friend constexpr sentinel_wrapper<ForwardIter> end(SizedForwardView& view) {
    return sentinel_wrapper<ForwardIter>{ForwardIter(view.ptr_ + 8)};
  }
  friend constexpr sentinel_wrapper<ForwardIter> end(SizedForwardView const& view) {
    return sentinel_wrapper<ForwardIter>{ForwardIter(view.ptr_ + 8)};
  }
};

constexpr auto operator-(sentinel_wrapper<ForwardIter> sent, ForwardIter iter) {
  return sent.base().base() - iter.base();
}
constexpr auto operator-(ForwardIter iter, sentinel_wrapper<ForwardIter> sent) {
  return iter.base() - sent.base().base();
}

template<class T>
concept SizeEnabled = requires(const std::ranges::common_view<T>& comm) {
  comm.size();
};

constexpr bool test() {
  int buffer[8] = {1, 2, 3, 4, 5, 6, 7, 8};

  {
    static_assert( SizeEnabled<SizedForwardView>);
    static_assert(!SizeEnabled<CopyableView>);
  }

  {
    std::ranges::common_view<SizedForwardView> common(SizedForwardView{buffer});
    assert(common.size() == 8);
  }

  {
    const std::ranges::common_view<SizedForwardView> common(SizedForwardView{buffer});
    assert(common.size() == 8);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}

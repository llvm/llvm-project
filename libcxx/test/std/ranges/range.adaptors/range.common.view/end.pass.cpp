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

// constexpr auto end();
// constexpr auto end() const requires range<const V>;

#include <ranges>
#include <cassert>

#include "test_macros.h"
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

using RandomAccessIter = random_access_iterator<int*>;
struct SizedRandomcAccessView : std::ranges::view_base {
  int *ptr_;
  constexpr SizedRandomcAccessView(int* ptr) : ptr_(ptr) {}
  friend constexpr RandomAccessIter begin(SizedRandomcAccessView& view) { return RandomAccessIter(view.ptr_); }
  friend constexpr RandomAccessIter begin(SizedRandomcAccessView const& view) { return RandomAccessIter(view.ptr_); }
  friend constexpr sentinel_wrapper<RandomAccessIter> end(SizedRandomcAccessView& view) {
    return sentinel_wrapper<RandomAccessIter>{RandomAccessIter(view.ptr_ + 8)};
  }
  friend constexpr sentinel_wrapper<RandomAccessIter> end(SizedRandomcAccessView const& view) {
    return sentinel_wrapper<RandomAccessIter>{RandomAccessIter(view.ptr_ + 8)};
  }
};

constexpr auto operator-(sentinel_wrapper<RandomAccessIter> sent, RandomAccessIter iter) {
  return sent.base().base() - iter.base();
}
constexpr auto operator-(RandomAccessIter iter, sentinel_wrapper<RandomAccessIter> sent) {
  return iter.base() - sent.base().base();
}

constexpr bool test() {
  int buffer[8] = {1, 2, 3, 4, 5, 6, 7, 8};

  {
    std::ranges::common_view<SizedRandomcAccessView> comm(SizedRandomcAccessView{buffer});
    assert(comm.end().base() == buffer + 8);
    // Note this should NOT be the sentinel type.
    ASSERT_SAME_TYPE(decltype(comm.end()), RandomAccessIter);
  }

  {
    const std::ranges::common_view<SizedRandomcAccessView> comm(SizedRandomcAccessView{buffer});
    assert(comm.end().base() == buffer + 8);
    // Note this should NOT be the sentinel type.
    ASSERT_SAME_TYPE(decltype(comm.end()), RandomAccessIter);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  int buffer[8] = {1, 2, 3, 4, 5, 6, 7, 8};

  using CommonForwardIter = std::common_iterator<ForwardIter, sentinel_wrapper<ForwardIter>>;
  using CommonIntIter = std::common_iterator<int*, sentinel_wrapper<int*>>;

  {
    std::ranges::common_view<SizedForwardView> comm(SizedForwardView{buffer});
    assert(comm.end() == CommonForwardIter(sentinel_wrapper<ForwardIter>(ForwardIter(buffer + 8))));
    ASSERT_SAME_TYPE(decltype(comm.end()), CommonForwardIter);
  }

  {
    std::ranges::common_view<CopyableView> comm(CopyableView{buffer});
    assert(comm.end() == CommonIntIter(sentinel_wrapper<int*>(buffer + 8)));
    ASSERT_SAME_TYPE(decltype(comm.end()), CommonIntIter);
  }

  {
    const std::ranges::common_view<SizedForwardView> comm(SizedForwardView{buffer});
    assert(comm.end() == CommonForwardIter(sentinel_wrapper<ForwardIter>(ForwardIter(buffer + 8))));
    ASSERT_SAME_TYPE(decltype(comm.end()), CommonForwardIter);
  }

  {
    const std::ranges::common_view<CopyableView> comm(CopyableView{buffer});
    assert(comm.end() == CommonIntIter(sentinel_wrapper<int*>(buffer + 8)));
    ASSERT_SAME_TYPE(decltype(comm.end()), CommonIntIter);
  }

  return 0;
}

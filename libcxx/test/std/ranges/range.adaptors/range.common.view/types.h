//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TEST_STD_RANGES_RANGE_ADAPTORS_RANGE_COMMON_VIEW_TYPES_H
#define TEST_STD_RANGES_RANGE_ADAPTORS_RANGE_COMMON_VIEW_TYPES_H

#include "test_iterators.h"

struct DefaultConstructibleView : std::ranges::view_base {
  int* begin_ = nullptr;
  int* end_ = nullptr;

  DefaultConstructibleView() = default;
  friend constexpr int* begin(DefaultConstructibleView& view) { return view.begin_; }
  friend constexpr int* begin(DefaultConstructibleView const& view) { return view.begin_; }
  friend constexpr sentinel_wrapper<int*> end(DefaultConstructibleView& view) {
    return sentinel_wrapper<int*>(view.end_);
  }
  friend constexpr sentinel_wrapper<int*> end(DefaultConstructibleView const& view) {
    return sentinel_wrapper<int*>(view.end_);
  }
};

struct ContiguousView : std::ranges::view_base {
  int* begin_;
  int* end_;

  constexpr ContiguousView(int* b, int* e) : begin_(b), end_(e) { }
  constexpr ContiguousView(ContiguousView&&) = default;
  constexpr ContiguousView& operator=(ContiguousView&&) = default;
  friend constexpr int* begin(ContiguousView& view) { return view.begin_; }
  friend constexpr int* begin(ContiguousView const& view) { return view.begin_; }
  friend constexpr sentinel_wrapper<int*> end(ContiguousView& view) {
    return sentinel_wrapper<int*>{view.end_};
  }
  friend constexpr sentinel_wrapper<int*> end(ContiguousView const& view) {
    return sentinel_wrapper<int*>{view.end_};
  }
};

struct CopyableView : std::ranges::view_base {
  int* begin_;
  int* end_;

  constexpr CopyableView(int* b, int* e) : begin_(b), end_(e) { }
  friend constexpr int* begin(CopyableView& view) { return view.begin_; }
  friend constexpr int* begin(CopyableView const& view) { return view.begin_; }
  friend constexpr sentinel_wrapper<int*> end(CopyableView& view) {
    return sentinel_wrapper<int*>{view.end_};
  }
  friend constexpr sentinel_wrapper<int*> end(CopyableView const& view) {
    return sentinel_wrapper<int*>{view.end_};
  }
};

using ForwardIter = forward_iterator<int*>;
struct SizedForwardView : std::ranges::view_base {
  int* begin_;
  int* end_;

  constexpr SizedForwardView(int* b, int* e) : begin_(b), end_(e) { }
  friend constexpr ForwardIter begin(SizedForwardView& view) { return ForwardIter(view.begin_); }
  friend constexpr ForwardIter begin(SizedForwardView const& view) { return ForwardIter(view.begin_); }
  friend constexpr sentinel_wrapper<ForwardIter> end(SizedForwardView& view) {
    return sentinel_wrapper<ForwardIter>{ForwardIter(view.end_)};
  }
  friend constexpr sentinel_wrapper<ForwardIter> end(SizedForwardView const& view) {
    return sentinel_wrapper<ForwardIter>{ForwardIter(view.end_)};
  }
};
// Required to make SizedForwardView a sized view.
constexpr auto operator-(sentinel_wrapper<ForwardIter> sent, ForwardIter iter) {
  return sent.base().base() - iter.base();
}
constexpr auto operator-(ForwardIter iter, sentinel_wrapper<ForwardIter> sent) {
  return iter.base() - sent.base().base();
}

using RandomAccessIter = random_access_iterator<int*>;
struct SizedRandomAccessView : std::ranges::view_base {
  int* begin_;
  int* end_;

  constexpr SizedRandomAccessView(int* b, int* e) : begin_(b), end_(e) { }
  friend constexpr RandomAccessIter begin(SizedRandomAccessView& view) { return RandomAccessIter(view.begin_); }
  friend constexpr RandomAccessIter begin(SizedRandomAccessView const& view) { return RandomAccessIter(view.begin_); }
  friend constexpr sentinel_wrapper<RandomAccessIter> end(SizedRandomAccessView& view) {
    return sentinel_wrapper<RandomAccessIter>{RandomAccessIter(view.end_)};
  }
  friend constexpr sentinel_wrapper<RandomAccessIter> end(SizedRandomAccessView const& view) {
    return sentinel_wrapper<RandomAccessIter>{RandomAccessIter(view.end_)};
  }
};
// Required to make SizedRandomAccessView a sized view.
constexpr auto operator-(sentinel_wrapper<RandomAccessIter> sent, RandomAccessIter iter) {
  return sent.base().base() - iter.base();
}
constexpr auto operator-(RandomAccessIter iter, sentinel_wrapper<RandomAccessIter> sent) {
  return iter.base() - sent.base().base();
}

#endif // TEST_STD_RANGES_RANGE_ADAPTORS_RANGE_COMMON_VIEW_TYPES_H

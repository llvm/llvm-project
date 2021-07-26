//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TEST_STD_RANGES_RANGE_ADAPTORS_RANGE_DROP_TYPES_H
#define TEST_STD_RANGES_RANGE_ADAPTORS_RANGE_DROP_TYPES_H

#include "test_macros.h"
#include "test_iterators.h"

int globalBuff[8];

struct ContiguousView : std::ranges::view_base {
  int start_;
  constexpr ContiguousView(int start = 0) : start_(start) {}
  constexpr ContiguousView(ContiguousView&&) = default;
  constexpr ContiguousView& operator=(ContiguousView&&) = default;
  friend constexpr int* begin(ContiguousView& view) { return globalBuff + view.start_; }
  friend constexpr int* begin(ContiguousView const& view) { return globalBuff + view.start_; }
  friend constexpr int* end(ContiguousView&) { return globalBuff + 8; }
  friend constexpr int* end(ContiguousView const&) { return globalBuff + 8; }
};

struct CopyableView : std::ranges::view_base {
  int start_;
  constexpr CopyableView(int start = 0) : start_(start) {}
  constexpr CopyableView(CopyableView const&) = default;
  constexpr CopyableView& operator=(CopyableView const&) = default;
  friend constexpr int* begin(CopyableView& view) { return globalBuff + view.start_; }
  friend constexpr int* begin(CopyableView const& view) { return globalBuff + view.start_; }
  friend constexpr int* end(CopyableView&) { return globalBuff + 8; }
  friend constexpr int* end(CopyableView const&) { return globalBuff + 8; }
};

using ForwardIter = forward_iterator<int*>;
struct ForwardView : std::ranges::view_base {
  constexpr ForwardView() = default;
  constexpr ForwardView(ForwardView&&) = default;
  constexpr ForwardView& operator=(ForwardView&&) = default;
  friend constexpr ForwardIter begin(ForwardView&) { return ForwardIter(globalBuff); }
  friend constexpr ForwardIter begin(ForwardView const&) { return ForwardIter(globalBuff); }
  friend constexpr ForwardIter end(ForwardView&) { return ForwardIter(globalBuff + 8); }
  friend constexpr ForwardIter end(ForwardView const&) { return ForwardIter(globalBuff + 8); }
};

struct ForwardRange {
  ForwardIter begin() const;
  ForwardIter end() const;
  ForwardIter begin();
  ForwardIter end();
};

struct ThrowingDefaultCtorForwardView : std::ranges::view_base {
  ThrowingDefaultCtorForwardView() noexcept(false);
  ForwardIter begin() const;
  ForwardIter end() const;
  ForwardIter begin();
  ForwardIter end();
};

struct NoDefaultCtorForwardView : std::ranges::view_base {
  NoDefaultCtorForwardView() = delete;
  ForwardIter begin() const;
  ForwardIter end() const;
  ForwardIter begin();
  ForwardIter end();
};

struct BorrowableRange {
  friend int* begin(BorrowableRange const& range);
  friend int* end(BorrowableRange const&);
  friend int* begin(BorrowableRange& range);
  friend int* end(BorrowableRange&);
};

template<>
inline constexpr bool std::ranges::enable_borrowed_range<BorrowableRange> = true;

struct BorrowableView : std::ranges::view_base {
  friend int* begin(BorrowableView const& range);
  friend int* end(BorrowableView const&);
  friend int* begin(BorrowableView& range);
  friend int* end(BorrowableView&);
};

template<>
inline constexpr bool std::ranges::enable_borrowed_range<BorrowableView> = true;

struct InputView : std::ranges::view_base {
  constexpr cpp20_input_iterator<int*> begin() const { return cpp20_input_iterator<int*>(globalBuff); }
  constexpr int* end() const { return globalBuff + 8; }
  constexpr cpp20_input_iterator<int*> begin() { return cpp20_input_iterator<int*>(globalBuff); }
  constexpr int* end() { return globalBuff + 8; }
};

constexpr bool operator==(const cpp20_input_iterator<int*> &lhs, int* rhs) { return lhs.base() == rhs; }
constexpr bool operator==(int* lhs, const cpp20_input_iterator<int*> &rhs) { return rhs.base() == lhs; }

struct Range {
  friend int* begin(Range const&);
  friend int* end(Range const&);
  friend int* begin(Range&);
  friend int* end(Range&);
};

using CountedIter = stride_counting_iterator<forward_iterator<int*>>;
struct CountedView : std::ranges::view_base {
  constexpr CountedIter begin() { return CountedIter(ForwardIter(globalBuff)); }
  constexpr CountedIter begin() const { return CountedIter(ForwardIter(globalBuff)); }
  constexpr CountedIter end() { return CountedIter(ForwardIter(globalBuff + 8)); }
  constexpr CountedIter end() const { return CountedIter(ForwardIter(globalBuff + 8)); }
};

#endif // TEST_STD_RANGES_RANGE_ADAPTORS_RANGE_DROP_TYPES_H

#ifndef TEST_STD_RANGES_RANGE_ADAPTORS_RANGE_TRANSFORM_TYPES_H
#define TEST_STD_RANGES_RANGE_ADAPTORS_RANGE_TRANSFORM_TYPES_H

#include "test_macros.h"
#include "test_iterators.h"
#include "test_range.h"

int globalBuff[8] = {0,1,2,3,4,5,6,7};

template<class T, class F>
concept ValidDropView = requires { typename std::ranges::transform_view<T, F>; };

struct ContiguousView : std::ranges::view_base {
  int start_;
  int *ptr_;
  constexpr ContiguousView(int* ptr = globalBuff, int start = 0) : start_(start), ptr_(ptr) {}
  constexpr ContiguousView(ContiguousView&&) = default;
  constexpr ContiguousView& operator=(ContiguousView&&) = default;
  friend constexpr int* begin(ContiguousView& view) { return view.ptr_ + view.start_; }
  friend constexpr int* begin(ContiguousView const& view) { return view.ptr_ + view.start_; }
  friend constexpr int* end(ContiguousView& view) { return view.ptr_ + 8; }
  friend constexpr int* end(ContiguousView const& view) { return view.ptr_ + 8; }
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
  int *ptr_;
  constexpr ForwardView(int* ptr = globalBuff) : ptr_(ptr) {}
  constexpr ForwardView(ForwardView&&) = default;
  constexpr ForwardView& operator=(ForwardView&&) = default;
  friend constexpr ForwardIter begin(ForwardView& view) { return ForwardIter(view.ptr_); }
  friend constexpr ForwardIter begin(ForwardView const& view) { return ForwardIter(view.ptr_); }
  friend constexpr ForwardIter end(ForwardView& view) { return ForwardIter(view.ptr_ + 8); }
  friend constexpr ForwardIter end(ForwardView const& view) { return ForwardIter(view.ptr_ + 8); }
};

using ForwardRange = test_common_range<forward_iterator>;

using RandomAccessIter = random_access_iterator<int*>;
struct RandomAccessView : std::ranges::view_base {
  RandomAccessIter begin() const noexcept;
  RandomAccessIter end() const noexcept;
  RandomAccessIter begin() noexcept;
  RandomAccessIter end() noexcept;
};

using BidirectionalIter = bidirectional_iterator<int*>;
struct BidirectionalView : std::ranges::view_base {
  BidirectionalIter begin() const;
  BidirectionalIter end() const;
  BidirectionalIter begin();
  BidirectionalIter end();
};

struct BorrowableRange {
  friend int* begin(BorrowableRange const& range);
  friend int* end(BorrowableRange const&);
  friend int* begin(BorrowableRange& range);
  friend int* end(BorrowableRange&);
};

template<>
inline constexpr bool std::ranges::enable_borrowed_range<BorrowableRange> = true;

struct InputView : std::ranges::view_base {
  int *ptr_;
  constexpr InputView(int* ptr = globalBuff) : ptr_(ptr) {}
  constexpr cpp20_input_iterator<int*> begin() const { return cpp20_input_iterator<int*>(ptr_); }
  constexpr int* end() const { return ptr_ + 8; }
  constexpr cpp20_input_iterator<int*> begin() { return cpp20_input_iterator<int*>(ptr_); }
  constexpr int* end() { return ptr_ + 8; }
};

constexpr bool operator==(const cpp20_input_iterator<int*> &lhs, int* rhs) { return lhs.base() == rhs; }
constexpr bool operator==(int* lhs, const cpp20_input_iterator<int*> &rhs) { return rhs.base() == lhs; }

struct SizedSentinelView : std::ranges::view_base {
  int count_;
  constexpr SizedSentinelView(int count = 8) : count_(count) {}
  constexpr RandomAccessIter begin() const { return RandomAccessIter(globalBuff); }
  constexpr int* end() const { return globalBuff + count_; }
  constexpr RandomAccessIter begin() { return RandomAccessIter(globalBuff); }
  constexpr int* end() { return globalBuff + count_; }
};

constexpr auto operator- (const RandomAccessIter &lhs, int* rhs) { return lhs.base() - rhs; }
constexpr auto operator- (int* lhs, const RandomAccessIter &rhs) { return lhs - rhs.base(); }
constexpr bool operator==(const RandomAccessIter &lhs, int* rhs) { return lhs.base() == rhs; }
constexpr bool operator==(int* lhs, const RandomAccessIter &rhs) { return rhs.base() == lhs; }

struct SizedSentinelNotConstView : std::ranges::view_base {
  ForwardIter begin() const;
  int *end() const;
  ForwardIter begin();
  int *end();
  size_t size();
};
bool operator==(const ForwardIter &lhs, int* rhs);
bool operator==(int* lhs, const ForwardIter &rhs);

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

using ThreeWayCompIter = three_way_contiguous_iterator<int*>;
struct ThreeWayCompView : std::ranges::view_base {
  constexpr ThreeWayCompIter begin() { return ThreeWayCompIter(globalBuff); }
  constexpr ThreeWayCompIter begin() const { return ThreeWayCompIter(globalBuff); }
  constexpr ThreeWayCompIter end() { return ThreeWayCompIter(globalBuff + 8); }
  constexpr ThreeWayCompIter end() const { return ThreeWayCompIter(globalBuff + 8); }
};

struct Increment {
  constexpr int operator()(int x) { return x + 1; }
};

struct IncrementConst {
  constexpr int operator()(int x) const { return x + 1; }
};

struct IncrementRef {
  constexpr int& operator()(int& x) { return ++x; }
};

struct IncrementRvalueRef {
  constexpr int&& operator()(int& x) { return std::move(++x); }
};

struct IncrementNoexcept {
  constexpr int operator()(int x) noexcept { return x + 1; }
};

#endif // TEST_STD_RANGES_RANGE_ADAPTORS_RANGE_TRANSFORM_TYPES_H

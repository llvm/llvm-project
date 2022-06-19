#ifndef TEST_STD_RANGES_RANGE_ADAPTORS_RANGE_TAKE_TYPES_H
#define TEST_STD_RANGES_RANGE_ADAPTORS_RANGE_TAKE_TYPES_H

#include <ranges>

#include "test_macros.h"
#include "test_iterators.h"
#include "test_range.h"

struct MoveOnlyView : std::ranges::view_base {
  int *ptr_;

  constexpr explicit MoveOnlyView(int* ptr) : ptr_(ptr) {}
  MoveOnlyView(MoveOnlyView&&) = default;
  MoveOnlyView& operator=(MoveOnlyView&&) = default;

  constexpr int* begin() const {return ptr_;}
  constexpr sentinel_wrapper<int*> end() const {return sentinel_wrapper<int*>{ptr_ + 8};}
};
static_assert( std::ranges::view<MoveOnlyView>);
static_assert( std::ranges::contiguous_range<MoveOnlyView>);
static_assert(!std::copyable<MoveOnlyView>);

struct CopyableView : std::ranges::view_base {
  int *ptr_;
  constexpr explicit CopyableView(int* ptr) : ptr_(ptr) {}

  constexpr int* begin() const {return ptr_;}
  constexpr sentinel_wrapper<int*> end() const {return sentinel_wrapper<int*>{ptr_ + 8};}
};
static_assert(std::ranges::view<CopyableView>);
static_assert(std::ranges::contiguous_range<CopyableView>);
static_assert(std::copyable<CopyableView>);

using ForwardIter = forward_iterator<int*>;
struct SizedForwardView : std::ranges::view_base {
  int *ptr_;
  constexpr explicit SizedForwardView(int* ptr) : ptr_(ptr) {}
  constexpr auto begin() const { return ForwardIter(ptr_); }
  constexpr auto end() const { return sized_sentinel<ForwardIter>(ForwardIter(ptr_ + 8)); }
};
static_assert(std::ranges::view<SizedForwardView>);
static_assert(std::ranges::forward_range<SizedForwardView>);
static_assert(std::ranges::sized_range<SizedForwardView>);

using RandomAccessIter = random_access_iterator<int*>;
struct SizedRandomAccessView : std::ranges::view_base {
  int *ptr_;
  constexpr explicit SizedRandomAccessView(int* ptr) : ptr_(ptr) {}
  constexpr auto begin() const { return RandomAccessIter(ptr_); }
  constexpr auto end() const { return sized_sentinel<RandomAccessIter>(RandomAccessIter(ptr_ + 8)); }
};
static_assert(std::ranges::view<SizedRandomAccessView>);
static_assert(std::ranges::random_access_range<SizedRandomAccessView>);
static_assert(std::ranges::sized_range<SizedRandomAccessView>);

#endif // TEST_STD_RANGES_RANGE_ADAPTORS_RANGE_TAKE_TYPES_H

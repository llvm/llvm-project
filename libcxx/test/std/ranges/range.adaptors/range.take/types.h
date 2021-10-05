#ifndef TEST_STD_RANGES_RANGE_ADAPTORS_RANGE_TAKE_TYPES_H
#define TEST_STD_RANGES_RANGE_ADAPTORS_RANGE_TAKE_TYPES_H

#include <ranges>

#include "test_macros.h"
#include "test_iterators.h"
#include "test_range.h"

struct ContiguousView : std::ranges::view_base {
  int *ptr_;

  constexpr explicit ContiguousView(int* ptr) : ptr_(ptr) {}
  ContiguousView(ContiguousView&&) = default;
  ContiguousView& operator=(ContiguousView&&) = default;

  constexpr int* begin() const {return ptr_;}
  constexpr sentinel_wrapper<int*> end() const {return sentinel_wrapper<int*>{ptr_ + 8};}
};
static_assert( std::ranges::view<ContiguousView>);
static_assert( std::ranges::contiguous_range<ContiguousView>);
static_assert(!std::copyable<ContiguousView>);

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
  constexpr auto end() const { return sentinel_wrapper<ForwardIter>(ForwardIter(ptr_ + 8)); }
};
// Required to make SizedForwardView a sized view.
constexpr auto operator-(sentinel_wrapper<ForwardIter> sent, ForwardIter iter) {
  return sent.base().base() - iter.base();
}
constexpr auto operator-(ForwardIter iter, sentinel_wrapper<ForwardIter> sent) {
  return iter.base() - sent.base().base();
}
static_assert(std::ranges::view<SizedForwardView>);
static_assert(std::ranges::forward_range<SizedForwardView>);
static_assert(std::ranges::sized_range<SizedForwardView>);

using RandomAccessIter = random_access_iterator<int*>;
struct SizedRandomAccessView : std::ranges::view_base {
  int *ptr_;
  constexpr explicit SizedRandomAccessView(int* ptr) : ptr_(ptr) {}
  constexpr auto begin() const { return RandomAccessIter(ptr_); }
  constexpr auto end() const { return sentinel_wrapper<RandomAccessIter>(RandomAccessIter(ptr_ + 8)); }
};
// Required to make SizedRandomAccessView a sized view.
constexpr auto operator-(sentinel_wrapper<RandomAccessIter> sent, RandomAccessIter iter) {
  return sent.base().base() - iter.base();
}
constexpr auto operator-(RandomAccessIter iter, sentinel_wrapper<RandomAccessIter> sent) {
  return iter.base() - sent.base().base();
}
static_assert(std::ranges::view<SizedRandomAccessView>);
static_assert(std::ranges::random_access_range<SizedRandomAccessView>);
static_assert(std::ranges::sized_range<SizedRandomAccessView>);

#endif // TEST_STD_RANGES_RANGE_ADAPTORS_RANGE_TAKE_TYPES_H

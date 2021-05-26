#ifndef TEST_STD_RANGES_RANGE_ADAPTORS_RANGE_TAKE_TYPES_H
#define TEST_STD_RANGES_RANGE_ADAPTORS_RANGE_TAKE_TYPES_H

#include <ranges>

#include "test_macros.h"
#include "test_iterators.h"
#include "test_range.h"

struct ContiguousView : std::ranges::view_base {
  int *ptr_;

  constexpr ContiguousView(int* ptr) : ptr_(ptr) {}
  ContiguousView(ContiguousView&&) = default;
  ContiguousView& operator=(ContiguousView&&) = default;

  constexpr int* begin() {return ptr_;}
  constexpr int* begin() const {return ptr_;}
  constexpr sentinel_wrapper<int*> end() {return sentinel_wrapper<int*>{ptr_ + 8};}
  constexpr sentinel_wrapper<int*> end() const {return sentinel_wrapper<int*>{ptr_ + 8};}
};

struct CopyableView : std::ranges::view_base {
  int *ptr_;
  constexpr CopyableView(int* ptr) : ptr_(ptr) {}

  constexpr int* begin() {return ptr_;}
  constexpr int* begin() const {return ptr_;}
  constexpr sentinel_wrapper<int*> end() {return sentinel_wrapper<int*>{ptr_ + 8};}
  constexpr sentinel_wrapper<int*> end() const {return sentinel_wrapper<int*>{ptr_ + 8};}
};

using ForwardIter = forward_iterator<int*>;
struct SizedForwardView : std::ranges::view_base {
  int *ptr_;
  constexpr SizedForwardView(int* ptr) : ptr_(ptr) {}
  constexpr friend ForwardIter begin(SizedForwardView& view) { return ForwardIter(view.ptr_); }
  constexpr friend ForwardIter begin(SizedForwardView const& view) { return ForwardIter(view.ptr_); }
  constexpr friend sentinel_wrapper<ForwardIter> end(SizedForwardView& view) {
    return sentinel_wrapper<ForwardIter>{ForwardIter(view.ptr_ + 8)};
  }
  constexpr friend sentinel_wrapper<ForwardIter> end(SizedForwardView const& view) {
    return sentinel_wrapper<ForwardIter>{ForwardIter(view.ptr_ + 8)};
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
  int *ptr_;
  constexpr SizedRandomAccessView(int* ptr) : ptr_(ptr) {}
  constexpr friend RandomAccessIter begin(SizedRandomAccessView& view) { return RandomAccessIter(view.ptr_); }
  constexpr friend RandomAccessIter begin(SizedRandomAccessView const& view) { return RandomAccessIter(view.ptr_); }
  constexpr friend sentinel_wrapper<RandomAccessIter> end(SizedRandomAccessView& view) {
    return sentinel_wrapper<RandomAccessIter>{RandomAccessIter(view.ptr_ + 8)};
  }
  constexpr friend sentinel_wrapper<RandomAccessIter> end(SizedRandomAccessView const& view) {
    return sentinel_wrapper<RandomAccessIter>{RandomAccessIter(view.ptr_ + 8)};
  }
};
// Required to make SizedRandomAccessView a sized view.
constexpr auto operator-(sentinel_wrapper<RandomAccessIter> sent, RandomAccessIter iter) {
  return sent.base().base() - iter.base();
}
constexpr auto operator-(RandomAccessIter iter, sentinel_wrapper<RandomAccessIter> sent) {
  return iter.base() - sent.base().base();
}

#endif // TEST_STD_RANGES_RANGE_ADAPTORS_RANGE_TAKE_TYPES_H

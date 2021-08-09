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

// constexpr auto begin();
// constexpr auto begin() const requires range<const V>;

#include <ranges>
#include <cassert>

#include "test_macros.h"
#include "test_iterators.h"
#include "test_range.h"

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

struct MutableView : std::ranges::view_base {
  int *ptr_;
  constexpr MutableView(int* ptr) : ptr_(ptr) {}
  constexpr int* begin() { return ptr_; }
  constexpr sentinel_wrapper<int*> end() { return sentinel_wrapper<int*>{ptr_ + 8}; }
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
  friend constexpr RandomAccessIter begin(SizedRandomAccessView& view) { return RandomAccessIter(view.ptr_); }
  friend constexpr RandomAccessIter begin(SizedRandomAccessView const& view) { return RandomAccessIter(view.ptr_); }
  friend constexpr sentinel_wrapper<RandomAccessIter> end(SizedRandomAccessView& view) {
    return sentinel_wrapper<RandomAccessIter>{RandomAccessIter(view.ptr_ + 8)};
  }
  friend constexpr sentinel_wrapper<RandomAccessIter> end(SizedRandomAccessView const& view) {
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

template<class T>
concept BeginEnabled = requires(const std::ranges::common_view<T>& comm) {
  comm.begin();
};

constexpr bool test() {
  int buffer[8] = {1, 2, 3, 4, 5, 6, 7, 8};

  {
    static_assert( BeginEnabled<CopyableView>);
    static_assert(!BeginEnabled<MutableView>);
  }

  {
    std::ranges::common_view<SizedRandomAccessView> comm(SizedRandomAccessView{buffer});
    assert(comm.begin() == begin(SizedRandomAccessView(buffer)));
    ASSERT_SAME_TYPE(decltype(comm.begin()), RandomAccessIter);
  }

  {
    const std::ranges::common_view<SizedRandomAccessView> comm(SizedRandomAccessView{buffer});
    assert(comm.begin() == begin(SizedRandomAccessView(buffer)));
    ASSERT_SAME_TYPE(decltype(comm.begin()), RandomAccessIter);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  // The non-constexpr tests:
  int buffer[8] = {1, 2, 3, 4, 5, 6, 7, 8};

  {
    std::ranges::common_view<SizedForwardView> comm(SizedForwardView{buffer});
    assert(comm.begin() == begin(SizedForwardView(buffer)));
    ASSERT_SAME_TYPE(decltype(comm.begin()), std::common_iterator<ForwardIter, sentinel_wrapper<ForwardIter>>);
  }

  {
    std::ranges::common_view<ContiguousView> comm(ContiguousView{buffer});
    assert(comm.begin() == begin(ContiguousView(buffer)));
    ASSERT_SAME_TYPE(decltype(comm.begin()), std::common_iterator<int*, sentinel_wrapper<int*>>);
  }

  {
    const std::ranges::common_view<SizedForwardView> comm(SizedForwardView{buffer});
    assert(comm.begin() == begin(SizedForwardView(buffer)));
    ASSERT_SAME_TYPE(decltype(comm.begin()), std::common_iterator<ForwardIter, sentinel_wrapper<ForwardIter>>);
  }

  {
    const std::ranges::common_view<CopyableView> comm(CopyableView{buffer});
    assert(comm.begin() == begin(CopyableView(buffer)));
    ASSERT_SAME_TYPE(decltype(comm.begin()), std::common_iterator<int*, sentinel_wrapper<int*>>);
  }

  return 0;
}

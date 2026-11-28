//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// UNSUPPORTED: c++03

// std::copy, std::move, std::copy_n, std::copy_backward, std::move_backward, and std::__unwrap_range unwrap bounded and
// user-defined contiguous iterators after validating the range once up front. Check that:
// - results are correct and preserve the original iterator's bounds (including in constant expressions),
// - bounded and user-defined contiguous iterators still lower to memmove without invoking operator* or operator++, and
// - non-sized input ranges, segmented iterators, and output iterators with void difference_type remain unaffected.

#include <__algorithm/unwrap_iter.h>
#include <__iterator/bounded_iter.h>
#include <__iterator/static_bounded_iter.h>
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <deque>
#include <iterator>
#include <list>
#include <memory>

#include "test_iterators.h"
#include "test_macros.h"

static_assert(!std::__unwrap_iter_may_strip_checks<int*>::value, "");
static_assert(!std::__unwrap_iter_may_strip_checks<std::__wrap_iter<int*> >::value, "");
static_assert(!std::__unwrap_iter_may_strip_checks<std::reverse_iterator<std::reverse_iterator<int*> > >::value, "");
static_assert(!std::__unwrap_iter_may_strip_checks<
                  std::reverse_iterator<std::reverse_iterator<std::__wrap_iter<int*> > > >::value,
              "");
static_assert(!std::__unwrap_iter_may_strip_checks<forward_iterator<int*> >::value, "");
static_assert(std::__unwrap_iter_may_strip_checks<std::__bounded_iter<int*> >::value, "");
static_assert(std::__unwrap_iter_may_strip_checks<std::__static_bounded_iter<int*, 4> >::value, "");
static_assert(std::__unwrap_iter_may_strip_checks<
                  std::reverse_iterator<std::reverse_iterator<std::__bounded_iter<int*> > > >::value,
              "");
#if TEST_STD_VER >= 20
static_assert(std::__unwrap_iter_may_strip_checks<contiguous_iterator<int*> >::value, "");
#endif

namespace {

// Output iterator whose difference_type is void, as with std::ostreambuf_iterator in C++17.
struct CountingOutputIterator {
  using iterator_category = std::output_iterator_tag;
  using value_type        = void;
  using difference_type   = void;
  using pointer           = void;
  using reference         = void;

  int* count;
  TEST_CONSTEXPR_CXX14 CountingOutputIterator& operator*() { return *this; }
  TEST_CONSTEXPR_CXX14 CountingOutputIterator& operator=(int) {
    ++*count;
    return *this;
  }
  TEST_CONSTEXPR_CXX14 CountingOutputIterator& operator++() { return *this; }
  TEST_CONSTEXPR_CXX14 CountingOutputIterator operator++(int) { return *this; }
};

#if TEST_STD_VER >= 20
// Fails to compile if std::copy falls back to an element-by-element loop instead of lowering to memmove.
template <class Iter>
class DisableDerefAndIncr : public Iter {
public:
  using Iter::Iter;
  constexpr DisableDerefAndIncr(const Iter& it) : Iter(it) {}
  void operator*()     = delete;
  void operator++()    = delete;
  void operator++(int) = delete;
};

// Pre-C++20 Cpp17InputIterator facade wrapping a non-random-access iterator while declaring an unconstrained
// non-template operator- whose body is ill-formed when instantiated, without specializing disable_sized_sentinel_for.
template <class Iter>
class UnconstrainedMinusFacadeIter {
  Iter base_{};

public:
  using iterator_category = typename std::iterator_traits<Iter>::iterator_category;
  using value_type        = typename std::iterator_traits<Iter>::value_type;
  using difference_type   = typename std::iterator_traits<Iter>::difference_type;
  using pointer           = typename std::iterator_traits<Iter>::pointer;
  using reference         = typename std::iterator_traits<Iter>::reference;

  constexpr UnconstrainedMinusFacadeIter() = default;
  constexpr explicit UnconstrainedMinusFacadeIter(Iter base) : base_(base) {}

  constexpr reference operator*() const { return *base_; }
  constexpr UnconstrainedMinusFacadeIter& operator++() {
    ++base_;
    return *this;
  }
  constexpr UnconstrainedMinusFacadeIter operator++(int) {
    auto tmp = *this;
    ++base_;
    return tmp;
  }
  friend constexpr bool operator==(const UnconstrainedMinusFacadeIter& a, const UnconstrainedMinusFacadeIter& b) {
    return a.base_ == b.base_;
  }
  friend constexpr bool operator!=(const UnconstrainedMinusFacadeIter& a, const UnconstrainedMinusFacadeIter& b) {
    return a.base_ != b.base_;
  }
  friend constexpr difference_type
  operator-(const UnconstrainedMinusFacadeIter& a, const UnconstrainedMinusFacadeIter& b) {
    return a.base_ - b.base_;
  }
};
#endif // TEST_STD_VER >= 20

} // namespace

#if TEST_STD_VER >= 20
template <class Iter>
struct std::pointer_traits<DisableDerefAndIncr<Iter>> : std::pointer_traits<Iter> {};
#endif

template <class SrcIter, class DstIter>
TEST_CONSTEXPR_CXX20 void check_copy_move(SrcIter src_first, SrcIter src_last, DstIter dst_first, DstIter dst_last) {
  // src holds {10, 20, 30, 40}; dst has room for 6 elements and starts at {1, 2, 3, 4, 5, 6}.
  DstIter out = std::copy(src_first, src_last, dst_first + 1);
  assert(out == dst_first + 5);
  assert(*dst_first == 1 && *(out - 4) == 10 && *out == 6);
  // Returned iterator preserves the original bounds of dst_first.
  assert(out - 5 == dst_first && out + 1 == dst_last);

  // Empty copy at both ends of bounds.
  assert(std::copy(src_first, src_first, dst_first) == dst_first);
  assert(std::copy(src_last, src_last, dst_last) == dst_last);
  assert(std::copy_backward(src_first, src_first, dst_first) == dst_first);
  assert(std::copy_backward(src_last, src_last, dst_last) == dst_last);

  // copy_n, move, overlapping copy/copy_backward, and move_backward.
  assert(std::copy_n(src_first, 3, dst_first) == dst_first + 3);
  assert(*dst_first == 10 && *(dst_first + 2) == 30);

  assert(std::move(src_first, src_last, dst_first + 2) == dst_last);
  assert(*(dst_first + 2) == 10 && *(dst_first + 5) == 40);

  assert(std::copy(dst_first + 2, dst_last, dst_first) == dst_first + 4);
  assert(*dst_first == 10 && *(dst_first + 3) == 40);

  assert(std::copy_backward(dst_first, dst_first + 4, dst_last) == dst_first + 2);
  assert(*(dst_first + 2) == 10 && *(dst_first + 5) == 40);

  assert(std::move_backward(src_first, src_last, dst_last - 1) == dst_first + 1);
  assert(*(dst_first + 1) == 10 && *(dst_first + 4) == 40);

#if TEST_STD_VER >= 20
  auto r_copy = std::ranges::copy(src_first, src_last, dst_first);
  assert(r_copy.in == src_last && r_copy.out == dst_first + 4);
  auto r_back = std::ranges::copy_backward(src_first, src_last, dst_last);
  assert(r_back.in == src_last && r_back.out == dst_first + 2);
  assert(std::ranges::equal(src_first, src_last, dst_first + 2, dst_last));
  assert(std::ranges::find(src_first, src_last, 30) == src_first + 2);
#endif
}

TEST_CONSTEXPR_CXX20 bool test() {
  const int src[4] = {10, 20, 30, 40};

  {
    int dst[6] = {1, 2, 3, 4, 5, 6};
    check_copy_move(std::__make_bounded_iter(src + 0, src + 0, src + 4),
                    std::__make_bounded_iter(src + 4, src + 0, src + 4),
                    std::__make_bounded_iter(dst + 0, dst + 0, dst + 6),
                    std::__make_bounded_iter(dst + 6, dst + 0, dst + 6));
  }
  {
    int dst[6] = {1, 2, 3, 4, 5, 6};
    check_copy_move(std::__make_static_bounded_iter<4>(src + 0, src + 0),
                    std::__make_static_bounded_iter<4>(src + 4, src + 0),
                    std::__make_static_bounded_iter<6>(dst + 0, dst + 0),
                    std::__make_static_bounded_iter<6>(dst + 6, dst + 0));
  }
  {
    int dst[6]      = {1, 2, 3, 4, 5, 6};
    using SrcBound  = std::__bounded_iter<const int*>;
    using DstBound  = std::__bounded_iter<int*>;
    using SrcRevRev = std::reverse_iterator<std::reverse_iterator<SrcBound> >;
    using DstRevRev = std::reverse_iterator<std::reverse_iterator<DstBound> >;

    check_copy_move(SrcRevRev(std::reverse_iterator<SrcBound>(std::__make_bounded_iter(src + 0, src + 0, src + 4))),
                    SrcRevRev(std::reverse_iterator<SrcBound>(std::__make_bounded_iter(src + 4, src + 0, src + 4))),
                    DstRevRev(std::reverse_iterator<DstBound>(std::__make_bounded_iter(dst + 0, dst + 0, dst + 6))),
                    DstRevRev(std::reverse_iterator<DstBound>(std::__make_bounded_iter(dst + 6, dst + 0, dst + 6))));
  }

  // Sub-range iterators into the same buffer where `last` has tighter end bounds than `first` (e.g. string_view::substr).
  {
    const int full[6] = {10, 20, 30, 40, 50, 60};
    int dst[3]        = {};
    auto first        = std::__make_bounded_iter(full + 0, full + 0, full + 6);
    auto prefix_end   = std::__make_bounded_iter(full + 3, full + 0, full + 3);
    assert(std::copy(first, prefix_end, std::__make_bounded_iter(dst + 0, dst + 0, dst + 3)) ==
           std::__make_bounded_iter(dst + 3, dst + 0, dst + 3));
    assert(dst[0] == 10 && dst[2] == 30);
  }

  // Output iterator with void difference_type.
  {
    int count = 0;
    std::copy(std::__make_bounded_iter(src + 0, src + 0, src + 4),
              std::__make_bounded_iter(src + 4, src + 0, src + 4),
              CountingOutputIterator{&count});
    assert(count == 4);
  }

#if TEST_STD_VER >= 20
  // Both __bounded_iter and user-defined contiguous_iterator lower to memmove without calling operator* or operator++.
  {
    int raw_in[4]        = {10, 20, 30, 40};
    int out1[4]          = {};
    using NoDerefBounded = DisableDerefAndIncr<std::__bounded_iter<int*>>;
    NoDerefBounded b_out = std::copy(
        NoDerefBounded(std::__make_bounded_iter(raw_in, raw_in, raw_in + 4)),
        NoDerefBounded(std::__make_bounded_iter(raw_in + 4, raw_in, raw_in + 4)),
        NoDerefBounded(std::__make_bounded_iter(out1, out1, out1 + 4)));
    assert(b_out == std::__make_bounded_iter(out1 + 4, out1, out1 + 4));
    assert(out1[0] == 10 && out1[3] == 40);

    int out2[4]       = {};
    using NoDerefCont = DisableDerefAndIncr<contiguous_iterator<int*>>;
    NoDerefCont c_out = std::copy(
        NoDerefCont(contiguous_iterator<int*>(raw_in)),
        NoDerefCont(contiguous_iterator<int*>(raw_in + 4)),
        NoDerefCont(contiguous_iterator<int*>(out2)));
    assert(c_out.base() == out2 + 4);
    assert(out2[0] == 10 && out2[3] == 40);
  }
#endif

  return true;
}

static void test_runtime_containers() {
  const std::deque<int> deq_src = {5, 6, 7};
  int deq_dst[4]                = {9, 9, 9, 9};
  assert(std::copy(deq_src.begin(), deq_src.end(), std::__make_bounded_iter(deq_dst, deq_dst, deq_dst + 4)) ==
         std::__make_bounded_iter(deq_dst + 3, deq_dst, deq_dst + 4));
  assert(deq_dst[0] == 5 && deq_dst[2] == 7 && deq_dst[3] == 9);

  std::deque<int> deq_out(4, 0);
  const int arr_src[3] = {15, 16, 17};
  assert(std::copy(std::__make_bounded_iter(arr_src, arr_src, arr_src + 3),
                   std::__make_bounded_iter(arr_src + 3, arr_src, arr_src + 3),
                   deq_out.begin()) == deq_out.begin() + 3);
  assert(deq_out[0] == 15 && deq_out[2] == 17 && deq_out[3] == 0);

  const std::list<int> list_src = {11, 12, 13};
  int list_dst[4]               = {9, 9, 9, 9};
  assert(std::copy(list_src.begin(), list_src.end(), std::__make_bounded_iter(list_dst, list_dst, list_dst + 4)) ==
         std::__make_bounded_iter(list_dst + 3, list_dst, list_dst + 4));
  assert(list_dst[0] == 11 && list_dst[2] == 13 && list_dst[3] == 9);

#if TEST_STD_VER >= 20
  int counted_dst[3] = {};
  auto counted_res   = std::ranges::copy(
      std::counted_iterator(list_src.begin(), 3),
      std::default_sentinel,
      std::__make_bounded_iter(counted_dst, counted_dst, counted_dst + 3));
  assert(counted_res.out == std::__make_bounded_iter(counted_dst + 3, counted_dst, counted_dst + 3));
  assert(counted_dst[0] == 11 && counted_dst[2] == 13);

  using FacadeListIter = UnconstrainedMinusFacadeIter<std::list<int>::const_iterator>;
  static_assert(std::sized_sentinel_for<FacadeListIter, FacadeListIter>);
  static_assert(!std::random_access_iterator<FacadeListIter>);
  int facade_dst[4] = {9, 9, 9, 9};
  assert(std::copy(FacadeListIter(list_src.begin()),
                   FacadeListIter(list_src.end()),
                   std::__make_bounded_iter(facade_dst, facade_dst, facade_dst + 4)) ==
         std::__make_bounded_iter(facade_dst + 3, facade_dst, facade_dst + 4));
  assert(facade_dst[0] == 11 && facade_dst[2] == 13 && facade_dst[3] == 9);
#endif
}

int main(int, char**) {
  test();
#if TEST_STD_VER >= 20
  static_assert(test());
#endif
  test_runtime_containers();
  return 0;
}

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: sanitizer-new-delete

// <algorithm>

// template<bidirectional_iterator I, sentinel_for<I> S, class Proj = identity,
//          indirect_unary_predicate<projected<I, Proj>> Pred>
//   requires permutable<I>
//   constexpr subrange<I>                                                         // constexpr since C++26
//     stable_partition(I first, S last, Pred pred, Proj proj = {});               // Since C++20
//
// template<bidirectional_range R, class Proj = identity,
//          indirect_unary_predicate<projected<iterator_t<R>, Proj>> Pred>
//   requires permutable<iterator_t<R>>
//   constexpr borrowed_subrange_t<R>                                              // constexpr since C++26
//     stable_partition(R&& r, Pred pred, Proj proj = {});                         // Since C++20

// [alg.partitions] requires stable_partition (without ExecutionPolicy) to apply the
// predicate exactly N = last - first times.

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <new>
#include <ranges>
#include <utility>
#include <vector>

#include "test_iterators.h"
#include "test_macros.h"

// Forces stable_partition onto its in-place (no temporary buffer) path. libc++
// allocates the buffer with ::operator new(..., nothrow), so returning nullptr
// makes the allocation fail. pair<int, int> is not over-aligned, so only the
// plain nothrow operator new needs to be replaced.
static bool g_nothrow_new_fails = false;

void* operator new(std::size_t n, const std::nothrow_t&) TEST_NOEXCEPT {
  return g_nothrow_new_fails ? nullptr : ::operator new(n);
}
void operator delete(void* p, const std::nothrow_t&) TEST_NOEXCEPT { ::operator delete(p); }

typedef std::pair<int, int> P; // (value, original index)

// stable_partition takes its predicate by value, so the count must
// exist outside the predicate object.
struct counting_proj {
  int* count_;
  int operator()(const P& p) const {
    ++*count_;
    return p.first;
  }
};
struct counting_is_even {
  int* count_;
  bool operator()(int v) const {
    ++*count_;
    return (v % 2) == 0;
  }
};

template <class Iter>
void test(std::size_t n) {
  // Worst case: every key is odd (predicate false) except the last, which is even
  // (predicate true). An implementation that evaluates the predicate before checking
  // for __first re-applies it to *__first, an element already known to be false,
  // once per node along the right segment, exceeding the required N applications.
  std::vector<P> data;
  data.reserve(n);
  for (std::size_t i = 0; i + 1 < n; ++i)
    data.push_back(P(1, static_cast<int>(i)));
  data.push_back(P(2, static_cast<int>(n - 1)));

  int pred_count        = 0;
  int proj_count        = 0;
  counting_is_even pred = {&pred_count};
  counting_proj proj    = {&proj_count};

  g_nothrow_new_fails = true;
  auto r              = std::ranges::stable_partition(Iter(data.data()), Iter(data.data() + n), pred, proj);
  g_nothrow_new_fails = false;

  // [alg.partitions]: exactly N applications of the predicate and projection.
  assert(static_cast<std::size_t>(pred_count) == n);
  assert(static_cast<std::size_t>(proj_count) == n);

  // Sanity check that the result is correctly and stably partitioned.
  assert(base(r.begin()) == data.data() + 1);
  assert(base(r.end()) == data.data() + n);
  assert(data[0].first % 2 == 0);
  for (std::size_t i = 1; i < n; ++i)
    assert(data[i].first % 2 != 0 && data[i].second == static_cast<int>(i - 1));
}

template <class Iter>
void test() {
  const std::size_t sizes[] = {1, 2, 3, 4, 5, 16, 17, 1000};
  for (std::size_t i = 0; i < sizeof(sizes) / sizeof(sizes[0]); ++i)
    test<Iter>(sizes[i]);
}

int main(int, char**) {
  test<P*>();
  test<bidirectional_iterator<P*> >();
  test<random_access_iterator<P*> >();
  return 0;
}

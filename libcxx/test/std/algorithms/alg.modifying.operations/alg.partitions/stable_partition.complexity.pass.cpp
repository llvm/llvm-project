//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: sanitizer-new-delete
// XFAIL: FROZEN-CXX03-HEADERS-FIXME

// <algorithm>

// template<BidirectionalIterator Iter, Predicate<auto, Iter::value_type> Pred>
//   requires ShuffleIterator<Iter>
//         && CopyConstructible<Pred>
//   constexpr Iter                                                               // constexpr since C++26
//   stable_partition(Iter first, Iter last, Pred pred);

// [alg.partitions] requires stable_partition (without ExecutionPolicy) to apply the
// predicate exactly N = last - first times.

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <new>
#include <utility>
#include <vector>

#include "test_iterators.h"
#include "test_macros.h"

// Forces stable_partition in-place (no temporary buffer). libc++ allocates
// the buffer with ::operator new(..., nothrow), so returning nullptr from
// it makes the allocation fail. pair<int, int> is not over-aligned, so only
// the plain nothrow operator new needs to be replaced.
static bool g_nothrow_new_fails = false;

void* operator new(std::size_t n, const std::nothrow_t&) TEST_NOEXCEPT {
  return g_nothrow_new_fails ? nullptr : ::operator new(n);
}
void operator delete(void* p, const std::nothrow_t&) TEST_NOEXCEPT { ::operator delete(p); }

typedef std::pair<int, int> P; // (value, original index)

// stable_partition takes its predicate by value, so the count must
// exist outside the predicate object.
struct counting_is_even {
  int* count_;
  bool operator()(const P& p) const {
    ++*count_;
    return (p.first % 2) == 0;
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

  int count             = 0;
  counting_is_even pred = {&count};

  g_nothrow_new_fails = true;
  Iter r              = std::stable_partition(Iter(data.data()), Iter(data.data() + n), pred);
  g_nothrow_new_fails = false;

  // [alg.partitions]: exactly N applications of the predicate.
  assert(static_cast<std::size_t>(count) == n);

  // Sanity check that the result is correctly and stably partitioned.
  assert(base(r) == data.data() + 1);
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

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
#include <cstdint>
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
  if (g_nothrow_new_fails)
    return nullptr;
#ifndef TEST_HAS_NO_EXCEPTIONS
  try {
    return ::operator new(n);
  } catch (...) {
    return nullptr;
  }
#else
  return ::operator new(n);
#endif
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

// Generate expected stable partition result for sanity check.
static std::vector<P> stable_reference(const std::vector<P>& in) {
  std::vector<P> out;
  out.reserve(in.size());
  for (std::size_t i = 0; i < in.size(); ++i)
    if (in[i].first % 2 == 0)
      out.push_back(in[i]);
  for (std::size_t i = 0; i < in.size(); ++i)
    if (in[i].first % 2 != 0)
      out.push_back(in[i]);
  return out;
}

// General case: deterministic pseudo-random keys.
static std::vector<P> make_shuffled(std::size_t n) {
  std::vector<P> data;
  data.reserve(n);
  std::uint64_t s = 0x10110010111001ULL;
  for (std::size_t i = 0; i < n; ++i) {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    data.push_back(P(static_cast<int>(s & 0xFF), static_cast<int>(i)));
  }
  return data;
}

// Worst case: every key is odd (predicate false) except the last, which is even
// (predicate true). An implementation that evaluates the predicate before checking
// for __first re-applies it to *__first, an element already known to be false,
// once per node along the right segment, exceeding the required N applications.
static std::vector<P> make_all_false_but_last(std::size_t n) {
  std::vector<P> data;
  data.reserve(n);
  for (std::size_t i = 0; i + 1 < n; ++i)
    data.push_back(P(1, static_cast<int>(i)));
  if (n > 0)
    data.push_back(P(2, static_cast<int>(n - 1)));
  return data;
}

template <class Iter>
void run_case(std::vector<P> data) {
  const std::size_t n           = data.size();
  const std::vector<P> expected = stable_reference(data);

  int count             = 0;
  counting_is_even pred = {&count};

  g_nothrow_new_fails = true;
  Iter r              = std::stable_partition(Iter(data.data()), Iter(data.data() + n), pred);
  g_nothrow_new_fails = false;

  // [alg.partitions]: exactly N applications of the predicate.
  assert(static_cast<std::size_t>(count) == n);

  // Sanity check that the result is correctly and stably partitioned, and the returned iterator
  // points at the first element for which the predicate is false.
  assert(data == expected);
  std::size_t num_true = 0;
  while (num_true < n && data[num_true].first % 2 == 0)
    ++num_true;
  assert(base(r) == data.data() + num_true);
}

template <class Iter>
void test() {
  const std::size_t sizes[] = {1, 2, 3, 4, 5, 6, 7, 8, 15, 16, 17, 63, 64, 65, 256, 1000, 4096};
  for (std::size_t i = 0; i < sizeof(sizes) / sizeof(sizes[0]); ++i) {
    run_case<Iter>(make_shuffled(sizes[i]));
    run_case<Iter>(make_all_false_but_last(sizes[i]));
  }
}

int main(int, char**) {
  test<P*>();
  test<bidirectional_iterator<P*> >();
  test<random_access_iterator<P*> >();
  return 0;
}

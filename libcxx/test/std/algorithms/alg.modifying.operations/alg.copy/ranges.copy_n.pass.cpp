//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-has-no-incomplete-ranges

// template<input_iterator I, weakly_incrementable O>
//   requires indirectly_copyable<I, O>
//   constexpr ranges::copy_n_result<I, O>
//     ranges::copy_n(I first, iter_difference_t<I> n, O result);

#include <algorithm>
#include <array>
#include <cassert>
#include <ranges>

#include "almost_satisfies_types.h"
#include "test_iterators.h"

template <class In, class Out = In, class Count = size_t>
concept HasCopyNIt = requires(In in, Count count, Out out) { std::ranges::copy_n(in, count, out); };

static_assert(HasCopyNIt<int*>);
static_assert(!HasCopyNIt<InputIteratorNotDerivedFrom>);
static_assert(!HasCopyNIt<InputIteratorNotIndirectlyReadable>);
static_assert(!HasCopyNIt<InputIteratorNotInputOrOutputIterator>);
static_assert(!HasCopyNIt<int*, WeaklyIncrementableNotMovable>);
struct NotIndirectlyCopyable {};
static_assert(!HasCopyNIt<int*, NotIndirectlyCopyable*>);
static_assert(!HasCopyNIt<int*, int*, SentinelForNotSemiregular>);
static_assert(!HasCopyNIt<int*, int*, SentinelForNotWeaklyEqualityComparableWith>);

static_assert(std::is_same_v<std::ranges::copy_result<int, long>, std::ranges::in_out_result<int, long>>);

template <class In, class Out, class Sent = In>
constexpr void test_iterators() {
  { // simple test
    std::array in {1, 2, 3, 4};
    std::array<int, 4> out;
    std::same_as<std::ranges::in_out_result<In, Out>> auto ret =
      std::ranges::copy_n(In(in.data()), in.size(), Out(out.data()));
    assert(in == out);
    assert(base(ret.in) == in.data() + in.size());
    assert(base(ret.out) == out.data() + out.size());
  }

  { // check that an empty range works
    std::array<int, 0> in;
    std::array<int, 0> out;
    auto ret = std::ranges::copy_n(In(in.data()), in.size(), Out(out.begin()));
    assert(base(ret.in) == in.data());
    assert(base(ret.out) == out.data());
  }
}

template <class Out>
constexpr void test_in_iterators() {
  test_iterators<cpp20_input_iterator<int*>, Out, sentinel_wrapper<cpp20_input_iterator<int*>>>();
  test_iterators<forward_iterator<int*>, Out>();
  test_iterators<bidirectional_iterator<int*>, Out>();
  test_iterators<random_access_iterator<int*>, Out>();
  test_iterators<contiguous_iterator<int*>, Out>();
}

template <class Out>
constexpr void test_proxy_in_iterators() {
  test_iterators<ProxyIterator<cpp20_input_iterator<int*>>, Out, sentinel_wrapper<ProxyIterator<cpp20_input_iterator<int*>>>>();
  test_iterators<ProxyIterator<forward_iterator<int*>>, Out>();
  test_iterators<ProxyIterator<bidirectional_iterator<int*>>, Out>();
  test_iterators<ProxyIterator<random_access_iterator<int*>>, Out>();
  test_iterators<ProxyIterator<contiguous_iterator<int*>>, Out>();
}

constexpr bool test() {
  test_in_iterators<cpp20_input_iterator<int*>>();
  test_in_iterators<forward_iterator<int*>>();
  test_in_iterators<bidirectional_iterator<int*>>();
  test_in_iterators<random_access_iterator<int*>>();
  test_in_iterators<contiguous_iterator<int*>>();

  test_proxy_in_iterators<ProxyIterator<cpp20_input_iterator<int*>>>();
  test_proxy_in_iterators<ProxyIterator<forward_iterator<int*>>>();
  test_proxy_in_iterators<ProxyIterator<bidirectional_iterator<int*>>>();
  test_proxy_in_iterators<ProxyIterator<random_access_iterator<int*>>>();
  test_proxy_in_iterators<ProxyIterator<contiguous_iterator<int*>>>();

  { // check that every element is copied exactly once
    struct CopyOnce {
      bool copied = false;
      constexpr CopyOnce() = default;
      constexpr CopyOnce(const CopyOnce& other) = delete;
      constexpr CopyOnce& operator=(const CopyOnce& other) {
        assert(!other.copied);
        copied = true;
        return *this;
      }
    };
    std::array<CopyOnce, 4> in {};
    std::array<CopyOnce, 4> out {};
    auto ret = std::ranges::copy_n(in.data(), in.size(), out.begin());
    assert(ret.in == in.end());
    assert(ret.out == out.end());
    assert(std::all_of(out.begin(), out.end(), [](const auto& e) { return e.copied; }));
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}

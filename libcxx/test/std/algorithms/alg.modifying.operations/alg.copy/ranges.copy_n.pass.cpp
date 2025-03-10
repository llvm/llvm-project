//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// UNSUPPORTED: c++03, c++11, c++14, c++17

// template<input_iterator I, weakly_incrementable O>
//   requires indirectly_copyable<I, O>
//   constexpr ranges::copy_n_result<I, O>
//     ranges::copy_n(I first, iter_difference_t<I> n, O result);

#include <algorithm>
#include <array>
#include <cassert>
#include <ranges>
#include <vector>

#include "almost_satisfies_types.h"
#include "test_macros.h"
#include "test_iterators.h"

template <class In, class Out = In, class Count = std::size_t>
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

template <class In, class Out>
constexpr void test_iterators() {
  { // simple test
    std::array in{1, 2, 3, 4};
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
    auto ret = std::ranges::copy_n(In(in.data()), in.size(), Out(out.data()));
    assert(base(ret.in) == in.data());
    assert(base(ret.out) == out.data());
  }
}

#if TEST_STD_VER >= 23
constexpr bool test_vector_bool(std::size_t N) {
  std::vector<bool> in(N, false);
  for (std::size_t i = 0; i < N; i += 2)
    in[i] = true;

  { // Test copy with aligned bytes
    std::vector<bool> out(N);
    std::ranges::copy_n(in.begin(), N, out.begin());
    assert(in == out);
  }
  { // Test copy with unaligned bytes
    std::vector<bool> out(N + 8);
    std::ranges::copy_n(in.begin(), N, out.begin() + 4);
    for (std::size_t i = 0; i < N; ++i)
      assert(out[i + 4] == in[i]);
  }

  return true;
};
#endif

constexpr bool test() {
  types::for_each(types::cpp20_input_iterator_list<int*>{}, []<class Out>() {
    types::for_each(types::cpp20_input_iterator_list<int*>{}, []<class In>() {
      test_iterators<In, Out>();
      test_iterators<ProxyIterator<In>, ProxyIterator<Out>>();
    });
  });

  { // check that every element is copied exactly once
    struct CopyOnce {
      bool copied                               = false;
      constexpr CopyOnce()                      = default;
      constexpr CopyOnce(const CopyOnce& other) = delete;
      constexpr CopyOnce& operator=(const CopyOnce& other) {
        assert(!other.copied);
        copied = true;
        return *this;
      }
    };
    std::array<CopyOnce, 4> in{};
    std::array<CopyOnce, 4> out{};
    auto ret = std::ranges::copy_n(in.begin(), in.size(), out.begin());
    assert(ret.in == in.end());
    assert(ret.out == out.end());
    assert(std::all_of(out.begin(), out.end(), [](const auto& e) { return e.copied; }));
  }

#if TEST_STD_VER >= 23
  { // Test vector<bool>::iterator optimization
    assert(test_vector_bool(8));
    assert(test_vector_bool(19));
    assert(test_vector_bool(32));
    assert(test_vector_bool(49));
    assert(test_vector_bool(64));
    assert(test_vector_bool(199));
    assert(test_vector_bool(256));
  }
#endif

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}

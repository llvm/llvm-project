//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// <algorithm>

// template<input_iterator I, sentinel_for<I> S, weakly_incrementable O, class Gen>
//   requires (forward_iterator<I> || random_access_iterator<O>) &&
//           indirectly_copyable<I, O> &&
//           uniform_random_bit_generator<remove_reference_t<Gen>>
//   O sample(I first, S last, O out, iter_difference_t<I> n, Gen&& g);                              // Since C++20
//
// template<input_range R, weakly_incrementable O, class Gen>
//   requires (forward_range<R> || random_access_iterator<O>) &&
//           indirectly_copyable<iterator_t<R>, O> &&
//           uniform_random_bit_generator<remove_reference_t<Gen>>
//   O sample(R&& r, O out, range_difference_t<R> n, Gen&& g);                                       // Since C++20

#include <algorithm>
#include <array>
#include <concepts>
#include <functional>
#include <random>
#include <ranges>
#include <utility>

#include "almost_satisfies_types.h"
#include "test_iterators.h"
#include "test_macros.h"

class RandGen {
public:
  constexpr static std::size_t min() { return 0; }
  constexpr static std::size_t max() { return 255; }

  constexpr std::size_t operator()() {
    flip = !flip;
    return flip;
  }

private:
  bool flip = false;
};

static_assert(std::uniform_random_bit_generator<RandGen>);
// `std::uniform_random_bit_generator` is a subset of requirements of `__libcpp_random_is_valid_urng`. Make sure that
// a type satisfying the required minimum is still accepted by `ranges::shuffle`.
LIBCPP_STATIC_ASSERT(!std::__libcpp_random_is_valid_urng<RandGen>::value);

struct BadGen {
  constexpr static std::size_t min() { return 255; }
  constexpr static std::size_t max() { return 0; }
  constexpr std::size_t operator()() const;
};
static_assert(!std::uniform_random_bit_generator<BadGen>);

// Test constraints of the (iterator, sentinel) overload.
// ======================================================

template <class Iter = int*, class Sent = int*, class Out = int*, class Gen = RandGen>
concept HasSampleIter =
    requires(Iter&& iter, Sent&& sent, Out&& out, std::iter_difference_t<Iter> n, Gen&& gen) {
      std::ranges::sample(std::forward<Iter>(iter), std::forward<Sent>(sent),
                          std::forward<Out>(out), n, std::forward<Gen>(gen));
    };

static_assert(HasSampleIter<int*, int*, int*, RandGen>);

// !input_iterator<I>
static_assert(!HasSampleIter<InputIteratorNotDerivedFrom>);
static_assert(!HasSampleIter<InputIteratorNotIndirectlyReadable>);
static_assert(!HasSampleIter<InputIteratorNotInputOrOutputIterator>);

// !sentinel_for<S, I>
static_assert(!HasSampleIter<int*, SentinelForNotSemiregular>);
static_assert(!HasSampleIter<int*, SentinelForNotWeaklyEqualityComparableWith>);

// !weakly_incrementable<O>
static_assert(!HasSampleIter<int*, int*, WeaklyIncrementableNotMovable>);

// (forward_iterator<I> || random_access_iterator<O>)
static_assert(HasSampleIter<
    forward_iterator<int*>, forward_iterator<int*>,
    cpp20_output_iterator<int*>
>);
static_assert(HasSampleIter<
    cpp20_input_iterator<int*>, sentinel_wrapper<cpp20_input_iterator<int*>>,
    random_access_iterator<int*>
>);
// !(forward_iterator<I> || random_access_iterator<O>)
static_assert(!HasSampleIter<
    cpp20_input_iterator<int*>, sentinel_wrapper<cpp20_input_iterator<int*>>,
    cpp20_output_iterator<int*>
>);

// !indirectly_copyable<I, O>
static_assert(!HasSampleIter<int*, int*, int**>);

// !uniform_random_bit_generator<remove_reference_t<Gen>>
static_assert(!HasSampleIter<int*, int*, int*, BadGen>);

// Test constraints of the (range) overload.
// =========================================

template <class Range, class Out = int*, class Gen = RandGen>
concept HasSampleRange =
    requires(Range&& range, Out&& out, std::ranges::range_difference_t<Range> n, Gen&& gen) {
      std::ranges::sample(std::forward<Range>(range), std::forward<Out>(out), n, std::forward<Gen>(gen));
    };

template <class T>
using R = UncheckedRange<T>;

static_assert(HasSampleRange<R<int*>, int*, RandGen>);

// !input_range<R>
static_assert(!HasSampleRange<InputRangeNotDerivedFrom>);
static_assert(!HasSampleRange<InputRangeNotIndirectlyReadable>);
static_assert(!HasSampleRange<InputRangeNotInputOrOutputIterator>);

// !weakly_incrementable<O>
static_assert(!HasSampleRange<R<int*>, WeaklyIncrementableNotMovable>);

// (forward_range<R> || random_access_iterator<O>)
static_assert(HasSampleRange<
    R<forward_iterator<int*>>,
    cpp20_output_iterator<int*>
>);
static_assert(HasSampleRange<
    R<cpp20_input_iterator<int*>>,
    random_access_iterator<int*>
>);
// !(forward_range<R> || random_access_iterator<O>)
static_assert(!HasSampleRange<
    R<cpp20_input_iterator<int*>>,
    cpp20_output_iterator<int*>
>);

// !indirectly_copyable<I, O>
static_assert(!HasSampleRange<R<int*>, int**>);

// !uniform_random_bit_generator<remove_reference_t<Gen>>
static_assert(!HasSampleRange<R<int*>, int*, BadGen>);

template <class Iter, class Sent, class Out, std::size_t N, class Gen>
void test_one(std::array<int, N> in, std::size_t n, Gen gen) {
  assert(n <= static_cast<std::size_t>(N));

  auto verify_is_subsequence = [&] (auto output) {
    auto sorted_input = in;
    std::ranges::sort(sorted_input);
    auto sorted_output = std::ranges::subrange(output.begin(), output.begin() + n);
    std::ranges::sort(sorted_output);
    assert(std::ranges::includes(sorted_input, sorted_output));
  };

  { // (iterator, sentinel) overload.
    auto begin = Iter(in.data());
    auto end = Sent(Iter(in.data() + in.size()));
    std::array<int, N> output;
    auto out = Out(output.begin());

    std::same_as<Out> decltype(auto) result = std::ranges::sample(
        std::move(begin), std::move(end), std::move(out), n, gen);
    assert(base(result) == output.data() + n);
    verify_is_subsequence(output);
    // The output of `sample` is implementation-specific.
  }

  { // (range) overload.
    auto begin = Iter(in.data());
    auto end = Sent(Iter(in.data() + in.size()));
    std::array<int, N> output;
    auto out = Out(output.begin());

    std::same_as<Out> decltype(auto) result = std::ranges::sample(std::ranges::subrange(
        std::move(begin), std::move(end)), std::move(out), n, gen);
    assert(base(result) == output.data() + n);
    verify_is_subsequence(output);
    // The output of `sample` is implementation-specific.
  }
}

template <class Iter, class Sent, class Out>
void test_iterators_iter_sent_out() {
  RandGen gen;

  // Empty sequence.
  test_one<Iter, Sent, Out, 0>({}, 0, gen);
  // 1-element sequence.
  test_one<Iter, Sent, Out, 1>({1}, 1, gen);
  // 2-element sequence.
  test_one<Iter, Sent, Out, 2>({1, 2}, 1, gen);
  test_one<Iter, Sent, Out, 2>({1, 2}, 2, gen);
  // n == 0.
  test_one<Iter, Sent, Out, 3>({1, 2, 3}, 0, gen);

  // Longer sequence.
  {
    std::array input = {1, 8, 2, 3, 4, 6, 5, 7};
    for (int i = 0; i <= static_cast<int>(input.size()); ++i){
      test_one<Iter, Sent, Out, input.size()>(input, i, gen);
    }
  }
}

template <class Iter, class Sent>
void test_iterators_iter_sent() {
  if constexpr (std::forward_iterator<Iter>) {
    test_iterators_iter_sent_out<Iter, Sent, cpp20_output_iterator<int*>>();
    test_iterators_iter_sent_out<Iter, Sent, forward_iterator<int*>>();
  }
  test_iterators_iter_sent_out<Iter, Sent, random_access_iterator<int*>>();
  test_iterators_iter_sent_out<Iter, Sent, contiguous_iterator<int*>>();
  test_iterators_iter_sent_out<Iter, Sent, int*>();
}

template <class Iter>
void test_iterators_iter() {
  if constexpr (std::sentinel_for<Iter, Iter>) {
    test_iterators_iter_sent<Iter, Iter>();
  }
  test_iterators_iter_sent<Iter, sentinel_wrapper<Iter>>();
}

void test_iterators() {
  test_iterators_iter<cpp20_input_iterator<int*>>();
  test_iterators_iter<random_access_iterator<int*>>();
  test_iterators_iter<contiguous_iterator<int*>>();
  test_iterators_iter<int*>();
  test_iterators_iter<const int*>();
}

// Checks the logic for wrapping the given iterator to make sure it works correctly regardless of the value category of
// the given generator object.
template <class Gen, bool CheckConst = true>
void test_generator() {
  std::array in = {1, 2, 3, 4, 5, 6, 7, 8};
  constexpr int N = 5;
  std::array<int, N> output;
  auto begin = in.begin();
  auto end = in.end();
  auto out = output.begin();

  { // Lvalue.
    Gen g;
    std::ranges::sample(begin, end, out, N, g);
    std::ranges::sample(in, out, N, g);
  }

  if constexpr (CheckConst) { // Const lvalue.
    const Gen g;
    std::ranges::sample(begin, end, out, N, g);
    std::ranges::sample(in, out, N, g);
  }

  { // Prvalue.
    std::ranges::sample(begin, end, out, N, Gen());
    std::ranges::sample(in, out, N, Gen());
  }

  { // Xvalue.
    Gen g1, g2;
    std::ranges::sample(begin, end, out, N, std::move(g1));
    std::ranges::sample(in, out, N, std::move(g2));
  }
}

// Checks the logic for wrapping the given iterator to make sure it works correctly regardless of whether the given
// generator class has a const or non-const invocation operator (or both).
void test_generators() {
  struct GenBase {
    constexpr static std::size_t min() { return 0; }
    constexpr static std::size_t max() { return 255; }
  };
  struct NonconstGen : GenBase {
    std::size_t operator()() { return 1; }
  };
  struct ConstGen : GenBase {
    std::size_t operator()() const { return 1; }
  };
  struct ConstAndNonconstGen : GenBase {
    std::size_t operator()() { return 1; }
    std::size_t operator()() const { return 1; }
  };

  test_generator<ConstGen>();
  test_generator<NonconstGen, /*CheckConst=*/false>();
  test_generator<ConstAndNonconstGen>();
}

void test() {
  test_iterators();
  test_generators();

  { // Stable (if `I` models `forward_iterator`).
    struct OrderedValue {
      int value;
      int original_order;
      bool operator==(const OrderedValue&) const = default;
      auto operator<=>(const OrderedValue& rhs) const { return value <=> rhs.value; }
    };

    const std::array<OrderedValue, 8> in = {{
      {1, 1}, {1, 2}, {1, 3}, {1, 4}, {1, 5}, {1, 6}, {1, 7}, {1, 8}
    }};

    { // (iterator, sentinel) overload.
      std::array<OrderedValue, in.size()> out;
      std::ranges::sample(in.begin(), in.end(), out.begin(), in.size(), RandGen());
      assert(out == in);
    }

    { // (range) overload.
      std::array<OrderedValue, in.size()> out;
      std::ranges::sample(in, out.begin(), in.size(), RandGen());
      assert(out == in);
    }
  }
}

int main(int, char**) {
  test();
  // Note: `ranges::sample` is not `constexpr`.

  return 0;
}

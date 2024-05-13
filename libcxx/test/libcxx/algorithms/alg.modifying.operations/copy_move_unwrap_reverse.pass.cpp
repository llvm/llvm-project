//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// <algorithm>

// These tests checks that `std::copy` and `std::move` (including their variations like `copy_n`) can unwrap multiple
// layers of reverse iterators.

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iterator>
#include <ranges>
#include <type_traits>

#include "test_iterators.h"

template <std::size_t N, class Iter>
requires (N == 0)
constexpr auto wrap_n_times(Iter i) {
  return i;
}

template <std::size_t N, class Iter>
requires (N != 0)
constexpr auto wrap_n_times(Iter i) {
  return std::make_reverse_iterator(wrap_n_times<N - 1>(i));
}

static_assert(std::is_same_v<decltype(wrap_n_times<2>(std::declval<int*>())),
                             std::reverse_iterator<std::reverse_iterator<int*>>>);

template <class InIter, template <class> class SentWrapper, class OutIter, std::size_t W1, size_t W2, class Func>
constexpr void test_one(Func func) {
  using From = std::iter_value_t<InIter>;
  using To = std::iter_value_t<OutIter>;

  const std::size_t N = 4;

  From input[N] = {{1}, {2}, {3}, {4}};
  To output[N];

  auto in     = wrap_n_times<W1>(InIter(input));
  auto in_end = wrap_n_times<W1>(InIter(input + N));
  auto sent   = SentWrapper<decltype(in_end)>(in_end);
  auto out    = wrap_n_times<W2>(OutIter(output));

  func(in, sent, out, N);

  assert(std::equal(input, input + N, output, [](const From& lhs, const To& rhs) {
        // Prevents warnings/errors due to mismatched signed-ness.
        return lhs == static_cast<From>(rhs);
    }));
}

template <class InIter, template <class> class SentWrapper, class OutIter, std::size_t W1, size_t W2>
constexpr void test_copy_and_move() {
  // Classic.
  if constexpr (std::same_as<InIter, SentWrapper<InIter>>) {
    test_one<InIter, SentWrapper, OutIter, W1, W2>([](auto first, auto last, auto out, std::size_t) {
      std::copy(first, last, out);
    });
    test_one<InIter, SentWrapper, OutIter, W1, W2>([](auto first, auto last, auto out, std::size_t n) {
      std::copy_backward(first, last, out + n);
    });
    test_one<InIter, SentWrapper, OutIter, W1, W2>([](auto first, auto, auto out, std::size_t n) {
      std::copy_n(first, n, out);
    });
    test_one<InIter, SentWrapper, OutIter, W1, W2>([](auto first, auto last, auto out, std::size_t) {
      std::move(first, last, out);
    });
    test_one<InIter, SentWrapper, OutIter, W1, W2>([](auto first, auto last, auto out, std::size_t n) {
      std::move_backward(first, last, out + n);
    });
  }

  // Ranges.
  test_one<InIter, SentWrapper, OutIter, W1, W2>([](auto first, auto last, auto out, std::size_t) {
    std::ranges::copy(first, last, out);
  });
  test_one<InIter, SentWrapper, OutIter, W1, W2>([](auto first, auto last, auto out, std::size_t n) {
    std::ranges::copy_backward(first, last, out + n);
  });
  test_one<InIter, SentWrapper, OutIter, W1, W2>([](auto first, auto, auto out, std::size_t n) {
    std::ranges::copy_n(first, n, out);
  });
  test_one<InIter, SentWrapper, OutIter, W1, W2>([](auto first, auto last, auto out, std::size_t) {
    std::ranges::move(first, last, out);
  });
  test_one<InIter, SentWrapper, OutIter, W1, W2>([](auto first, auto last, auto out, std::size_t n) {
    std::ranges::move_backward(first, last, out + n);
  });
}

template <std::size_t W1, size_t W2, class From, class To, template <class> class SentWrapper>
constexpr void test_all_permutations_with_counts_from_to_sent() {
  test_copy_and_move<From*, SentWrapper, To*, W1, W2>();
  test_copy_and_move<contiguous_iterator<From*>, SentWrapper, To*, W1, W2>();
  test_copy_and_move<From*, SentWrapper, contiguous_iterator<To*>, W1, W2>();
  test_copy_and_move<contiguous_iterator<From*>, SentWrapper, contiguous_iterator<To*>, W1, W2>();

  if (!std::same_as<From, To>) {
    test_copy_and_move<To*, SentWrapper, From*, W1, W2>();
    test_copy_and_move<contiguous_iterator<To*>, SentWrapper, From*, W1, W2>();
    test_copy_and_move<To*, SentWrapper, contiguous_iterator<From*>, W1, W2>();
    test_copy_and_move<contiguous_iterator<To*>, SentWrapper, contiguous_iterator<From*>, W1, W2>();
  }
}

template <std::size_t W1, size_t W2>
constexpr void test_all_permutations_with_counts() {
  test_all_permutations_with_counts_from_to_sent<W1, W2, int, int, std::type_identity_t>();
  test_all_permutations_with_counts_from_to_sent<W1, W2, int, int, sized_sentinel>();
  test_all_permutations_with_counts_from_to_sent<W1, W2, std::int32_t, std::uint32_t, std::type_identity_t>();
  test_all_permutations_with_counts_from_to_sent<W1, W2, std::int32_t, std::uint32_t, sized_sentinel>();
}

constexpr bool test() {
  test_all_permutations_with_counts<0, 0>();
  test_all_permutations_with_counts<0, 2>();
  test_all_permutations_with_counts<2, 0>();
  test_all_permutations_with_counts<2, 2>();
  test_all_permutations_with_counts<2, 4>();
  test_all_permutations_with_counts<4, 4>();

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// When the debug mode is enabled, we don't unwrap iterators in `std::copy` and similar algorithms so we don't get this
// optimization.
// UNSUPPORTED: libcpp-has-debug-mode
// In the modules build, adding another overload of `memmove` doesn't work.
// UNSUPPORTED: modules-build
// GCC complains about "ambiguating" `__builtin_memmove`.
// UNSUPPORTED: gcc

// <algorithm>

// These tests checks that `std::copy` and `std::move` (including their variations like `copy_n`) forward to
// `memmove` when possible.

#include <cstddef>

struct Foo {
  int i = 0;

  Foo() = default;
  Foo(int set_i) : i(set_i) {}

  friend bool operator==(const Foo&, const Foo&) = default;
};

static bool memmove_called = false;

// This overload is a better match than the actual `builtin_memmove`, so it should hijack the call inside `std::copy`
// and similar algorithms.
void* __builtin_memmove(Foo* dst, Foo* src, size_t count) {
  memmove_called = true;
  return __builtin_memmove(static_cast<void*>(dst), static_cast<void*>(src), count);
}

#include <algorithm>
#include <cassert>
#include <iterator>
#include <ranges>
#include <type_traits>

#include "test_iterators.h"

static_assert(std::is_trivially_copyable_v<Foo>);

template <size_t N, class Iter>
requires (N == 0)
constexpr auto wrap_n_times(Iter i) {
  return i;
}

template <size_t N, class Iter>
requires (N != 0)
constexpr auto wrap_n_times(Iter i) {
  return std::make_reverse_iterator(wrap_n_times<N - 1>(i));
}

static_assert(std::is_same_v<decltype(wrap_n_times<2>(std::declval<int*>())),
                             std::reverse_iterator<std::reverse_iterator<int*>>>);

template <class InIter, template <class> class SentWrapper, class OutIter, size_t W1, size_t W2, class Func>
void test_one(Func func) {
  {
    const size_t N = 4;

    Foo input[N] = {{1}, {2}, {3}, {4}};
    Foo output[N];

    auto in     = wrap_n_times<W1>(InIter(input));
    auto in_end = wrap_n_times<W1>(InIter(input + N));
    auto sent   = SentWrapper<decltype(in_end)>(in_end);
    auto out    = wrap_n_times<W2>(OutIter(output));

    assert(!memmove_called);
    func(in, sent, out, N);

    assert(std::equal(input, input + N, output));
    assert(memmove_called);
    memmove_called = false;
  }

  {
    const size_t N = 0;

    Foo input[1]  = {1};
    Foo output[1] = {2};

    auto in     = wrap_n_times<W1>(InIter(input));
    auto in_end = wrap_n_times<W1>(InIter(input + N));
    auto sent   = SentWrapper<decltype(in_end)>(in_end);
    auto out    = wrap_n_times<W2>(OutIter(output));

    assert(!memmove_called);
    func(in, sent, out, N);

    assert(output[0] == 2);
    assert(memmove_called);
    memmove_called = false;
  }
}

template <class InIter, template <class> class SentWrapper, class OutIter, size_t W1, size_t W2>
void test_copy_and_move() {
  // Classic.
  if constexpr (std::same_as<InIter, SentWrapper<InIter>>) {
    test_one<InIter, SentWrapper, OutIter, W1, W2>([](auto first, auto last, auto out, size_t) {
      std::copy(first, last, out);
    });
    test_one<InIter, SentWrapper, OutIter, W1, W2>([](auto first, auto last, auto out, size_t n) {
      std::copy_backward(first, last, out + n);
    });
    test_one<InIter, SentWrapper, OutIter, W1, W2>([](auto first, auto, auto out, size_t n) {
      std::copy_n(first, n, out);
    });
    test_one<InIter, SentWrapper, OutIter, W1, W2>([](auto first, auto last, auto out, size_t) {
      std::move(first, last, out);
    });
    test_one<InIter, SentWrapper, OutIter, W1, W2>([](auto first, auto last, auto out, size_t n) {
      std::move_backward(first, last, out + n);
    });
  }

  // Ranges.
  test_one<InIter, SentWrapper, OutIter, W1, W2>([](auto first, auto last, auto out, size_t) {
    std::ranges::copy(first, last, out);
  });
  test_one<InIter, SentWrapper, OutIter, W1, W2>([](auto first, auto last, auto out, size_t n) {
    std::ranges::copy_backward(first, last, out + n);
  });
  test_one<InIter, SentWrapper, OutIter, W1, W2>([](auto first, auto, auto out, size_t n) {
    std::ranges::copy_n(first, n, out);
  });
  test_one<InIter, SentWrapper, OutIter, W1, W2>([](auto first, auto last, auto out, size_t) {
    std::ranges::move(first, last, out);
  });
  test_one<InIter, SentWrapper, OutIter, W1, W2>([](auto first, auto last, auto out, size_t n) {
    std::ranges::move_backward(first, last, out + n);
  });
}

template <class InIter, template <class> class SentWrapper, class OutIter>
void test_all_permutations_with_initer_sent_outiter() {
  test_copy_and_move<InIter, SentWrapper, OutIter, 0, 0>();
  test_copy_and_move<InIter, SentWrapper, OutIter, 0, 2>();
  test_copy_and_move<InIter, SentWrapper, OutIter, 2, 0>();
  test_copy_and_move<InIter, SentWrapper, OutIter, 2, 2>();
  test_copy_and_move<InIter, SentWrapper, OutIter, 2, 4>();
  test_copy_and_move<InIter, SentWrapper, OutIter, 4, 4>();
}

template <class InIter, template <class> class SentWrapper>
void test_all_permutations_with_initer_sent() {
  test_all_permutations_with_initer_sent_outiter<InIter, SentWrapper, Foo*>();
  test_all_permutations_with_initer_sent_outiter<InIter, SentWrapper, contiguous_iterator<Foo*>>();
}

template <class InIter>
void test_all_permutations_with_initer() {
  test_all_permutations_with_initer_sent<InIter, std::type_identity_t>();
  test_all_permutations_with_initer_sent<InIter, sized_sentinel>();
}

void test() {
  test_all_permutations_with_initer<Foo*>();
  test_all_permutations_with_initer<contiguous_iterator<Foo*>>();
}

int main(int, char**) {
  test();
  // The test relies on a global variable, so it cannot be made `constexpr`; the `memmove` optimization is not used in
  // `constexpr` mode anyway.

  return 0;
}

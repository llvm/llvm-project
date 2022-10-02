//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// <algorithm>

// These tests checks that `std::copy` and `std::move` (including their variations like `copy_n`) don't forward to
// `std::memmove` when doing so would be observable.

#include <algorithm>
#include <cassert>
#include <iterator>
#include <ranges>
#include <type_traits>

#include "test_iterators.h"
#include "test_macros.h"

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

struct NonTrivialMoveAssignment {
  int i;

  constexpr NonTrivialMoveAssignment() = default;
  constexpr NonTrivialMoveAssignment(int set_i) : i(set_i) {}

  constexpr NonTrivialMoveAssignment(NonTrivialMoveAssignment&& rhs) = default;
  constexpr NonTrivialMoveAssignment& operator=(NonTrivialMoveAssignment&& rhs) noexcept {
    i = rhs.i;
    return *this;
  }

  constexpr friend bool operator==(const NonTrivialMoveAssignment&, const NonTrivialMoveAssignment&) = default;
};

static_assert(!std::is_trivially_move_assignable_v<NonTrivialMoveAssignment>);

struct NonTrivialMoveCtr {
  int i;

  constexpr NonTrivialMoveCtr() = default;
  constexpr NonTrivialMoveCtr(int set_i) : i(set_i) {}

  constexpr NonTrivialMoveCtr(NonTrivialMoveCtr&& rhs) noexcept : i(rhs.i) {}
  constexpr NonTrivialMoveCtr& operator=(NonTrivialMoveCtr&& rhs) = default;

  constexpr friend bool operator==(const NonTrivialMoveCtr&, const NonTrivialMoveCtr&) = default;
};

static_assert(std::is_trivially_move_assignable_v<NonTrivialMoveCtr>);
static_assert(!std::is_trivially_copyable_v<NonTrivialMoveCtr>);

struct NonTrivialCopyAssignment {
  int i;

  constexpr NonTrivialCopyAssignment() = default;
  constexpr NonTrivialCopyAssignment(int set_i) : i(set_i) {}

  constexpr NonTrivialCopyAssignment(const NonTrivialCopyAssignment& rhs) = default;
  constexpr NonTrivialCopyAssignment& operator=(const NonTrivialCopyAssignment& rhs) {
    i = rhs.i;
    return *this;
  }

  constexpr friend bool operator==(const NonTrivialCopyAssignment&, const NonTrivialCopyAssignment&) = default;
};

static_assert(!std::is_trivially_copy_assignable_v<NonTrivialCopyAssignment>);

struct NonTrivialCopyCtr {
  int i;

  constexpr NonTrivialCopyCtr() = default;
  constexpr NonTrivialCopyCtr(int set_i) : i(set_i) {}

  constexpr NonTrivialCopyCtr(const NonTrivialCopyCtr& rhs) : i(rhs.i) {}
  constexpr NonTrivialCopyCtr& operator=(const NonTrivialCopyCtr& rhs) = default;

  constexpr friend bool operator==(const NonTrivialCopyCtr&, const NonTrivialCopyCtr&) = default;
};

static_assert(std::is_trivially_copy_assignable_v<NonTrivialCopyCtr>);
static_assert(!std::is_trivially_copyable_v<NonTrivialCopyCtr>);

// Unwrapping the iterator inside `std::copy` and similar algorithms relies on `to_address`. If the `memmove`
// optimization is used, the result of the call to `to_address` will be passed to `memmove`. This test deliberately
// specializes `to_address` for `contiguous_iterator` to return a type that doesn't implicitly convert to `void*`, so
// that a call to `memmove` would fail to compile.
template <>
struct std::pointer_traits<::contiguous_iterator<NonTrivialCopyAssignment*>> {
  static constexpr ::contiguous_iterator<NonTrivialCopyAssignment*>
  to_address(const ::contiguous_iterator<NonTrivialCopyAssignment*>& iter) {
    return iter;
  }
};
template <>
struct std::pointer_traits<::contiguous_iterator<NonTrivialMoveAssignment*>> {
  static constexpr ::contiguous_iterator<NonTrivialMoveAssignment*>
  to_address(const ::contiguous_iterator<NonTrivialMoveAssignment*>& iter) {
    return iter;
  }
};

template <class InIter, template <class> class SentWrapper, class OutIter, size_t W1, size_t W2, class Func>
constexpr void test_one(Func func) {
  using Value = typename std::iterator_traits<InIter>::value_type;

  {
    const size_t N = 4;

    Value input[N] = {Value{1}, {2}, {3}, {4}};
    Value output[N];

    auto in     = wrap_n_times<W1>(InIter(input));
    auto in_end = wrap_n_times<W1>(InIter(input + N));
    auto sent   = SentWrapper<decltype(in_end)>(in_end);
    auto out    = wrap_n_times<W2>(OutIter(output));

    func(in, sent, out, N);
    assert(std::equal(input, input + N, output));
  }

  {
    const size_t N = 0;

    Value input[1]  = {1};
    Value output[1] = {2};

    auto in     = wrap_n_times<W1>(InIter(input));
    auto in_end = wrap_n_times<W1>(InIter(input + N));
    auto sent   = SentWrapper<decltype(in_end)>(in_end);
    auto out    = wrap_n_times<W2>(OutIter(output));

    func(in, sent, out, N);
    assert(output[0] == Value(2));
  }
}

template <class InIter, template <class> class SentWrapper, class OutIter, size_t W1, size_t W2>
constexpr void test_copy() {
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
}

template <class InIter, template <class> class SentWrapper, class OutIter, size_t W1, size_t W2>
constexpr void test_move() {
  if constexpr (std::same_as<InIter, SentWrapper<InIter>>) {
    test_one<InIter, SentWrapper, OutIter, W1, W2>([](auto first, auto last, auto out, size_t) {
      std::move(first, last, out);
    });
    test_one<InIter, SentWrapper, OutIter, W1, W2>([](auto first, auto last, auto out, size_t n) {
      std::move_backward(first, last, out + n);
    });
  }

  // Ranges.
  test_one<InIter, SentWrapper, OutIter, W1, W2>([](auto first, auto last, auto out, size_t) {
    std::ranges::move(first, last, out);
  });
  test_one<InIter, SentWrapper, OutIter, W1, W2>([](auto first, auto last, auto out, size_t n) {
    std::ranges::move_backward(first, last, out + n);
  });
}

template <class T, size_t W1, size_t W2>
constexpr void test_copy_with_type() {
  using CopyIter = contiguous_iterator<T*>;

  test_copy<CopyIter, std::type_identity_t, CopyIter, W1, W2>();
  test_copy<CopyIter, sized_sentinel, CopyIter, W1, W2>();
  test_copy<CopyIter, std::type_identity_t, T*, W1, W2>();
  test_copy<T*, std::type_identity_t, CopyIter, W1, W2>();
}

template <class T, size_t W1, size_t W2>
constexpr void test_move_with_type() {
  using MoveIter = contiguous_iterator<T*>;

  test_move<MoveIter, std::type_identity_t, MoveIter, W1, W2>();
  test_move<MoveIter, sized_sentinel, MoveIter, W1, W2>();
  test_move<MoveIter, std::type_identity_t, T*, W1, W2>();
  test_move<T*, std::type_identity_t, MoveIter, W1, W2>();
}

template <size_t W1, size_t W2>
constexpr void test_copy_and_move() {
  test_copy_with_type<NonTrivialCopyAssignment, W1, W2>();
  test_copy_with_type<NonTrivialCopyCtr, W1, W2>();

  test_move_with_type<NonTrivialMoveAssignment, W1, W2>();
  test_move_with_type<NonTrivialMoveCtr, W1, W2>();
}

constexpr bool test() {
  test_copy_and_move<0, 0>();
  test_copy_and_move<0, 2>();
  test_copy_and_move<2, 0>();
  test_copy_and_move<2, 2>();
  test_copy_and_move<2, 4>();
  test_copy_and_move<4, 4>();

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// When the debug mode is enabled, we don't unwrap iterators in `std::copy` and similar algorithms so we never get the
// optimization.
// UNSUPPORTED: libcpp-has-debug-mode
// In the modules build, adding another overload of `memmove` doesn't work.
// UNSUPPORTED: modules-build
// GCC complains about "ambiguating" `__builtin_memmove`.
// UNSUPPORTED: gcc

// <algorithm>

#include <cassert>
#include <cstddef>

// These tests check that `std::copy` and `std::move` (including their variations like `copy_n`) don't forward to
// `std::memmove` when doing so would be observable.

// This template is a better match than the actual `builtin_memmove` (it can match the pointer type exactly, without an
// implicit conversion to `void*`), so it should hijack the call inside `std::copy` and similar algorithms if it's made.
template <class Dst, class Src>
constexpr void* __builtin_memmove(Dst*, Src*, size_t) {
  assert(false);
  return nullptr;
}

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iterator>
#include <ranges>
#include <type_traits>

#include "test_iterators.h"
#include "test_macros.h"

// S1 and S2 are simple structs that are convertible to each other and have the same bit representation.
struct S1 {
  int x;

  constexpr S1() = default;
  constexpr S1(int set_x) : x(set_x) {}

  friend constexpr bool operator==(const S1& lhs, const S1& rhs) { return lhs.x == rhs.x; }
};

struct S2 {
  int x;

  constexpr S2() = default;
  constexpr S2(int set_x) : x(set_x) {}
  constexpr S2(S1 from) : x(from.x) {}

  friend constexpr bool operator==(const S1& lhs, const S2& rhs) { return lhs.x == rhs.x; }
  friend constexpr bool operator==(const S2& lhs, const S2& rhs) { return lhs.x == rhs.x; }
};

// U1 and U2 are simple unions that are convertible to each other and have the same bit representation.
union U1 {
  int x;

  constexpr U1() = default;
  constexpr U1(int set_x) : x(set_x) {}

  friend constexpr bool operator==(const U1& lhs, const U1& rhs) { return lhs.x == rhs.x; }
};

union U2 {
  int x;

  constexpr U2() = default;
  constexpr U2(int set_x) : x(set_x) {}
  constexpr U2(U1 from) : x(from.x) {}

  friend constexpr bool operator==(const U1& lhs, const U2& rhs) { return lhs.x == rhs.x; }
  friend constexpr bool operator==(const U2& lhs, const U2& rhs) { return lhs.x == rhs.x; }
};

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
static_assert(!std::is_trivially_assignable<NonTrivialMoveAssignment&, NonTrivialMoveAssignment&>::value);

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

template <class T>
constexpr T make(int from) {
  return T(from);
}

template <typename PtrT, typename T = std::remove_pointer_t<PtrT>>
static T make_internal_array[5] = {T(), T(), T(), T(), T()};

template <class T>
requires std::is_pointer_v<T>
constexpr T make(int i) {
  if constexpr (!std::same_as<std::remove_pointer_t<T>, void>) {
    return make_internal_array<T> + i;
  } else {
    return make_internal_array<int> + i;
  }
}

template <class InIter, template <class> class SentWrapper, class OutIter, class Func>
constexpr void test_one(Func func) {
  using From = typename std::iterator_traits<InIter>::value_type;
  using To = typename std::iterator_traits<OutIter>::value_type;

  {
    const size_t N = 5;

    From input[N] = {make<From>(0), make<From>(1), make<From>(2), make<From>(3), make<From>(4)};
    To output[N];

    auto in     = InIter(input);
    auto in_end = InIter(input + N);
    auto sent   = SentWrapper<decltype(in_end)>(in_end);
    auto out    = OutIter(output);

    func(in, sent, out, N);
    if constexpr (!std::same_as<To, bool>) {
      assert(std::equal(input, input + N, output));
    } else {
      bool expected[N] = {false, true, true, true, true};
      assert(std::equal(output, output + N, expected));
    }
  }

  {
    const size_t N = 0;

    From input[1]  = {make<From>(1)};
    To output[1] = {make<To>(2)};

    auto in     = InIter(input);
    auto in_end = InIter(input + N);
    auto sent   = SentWrapper<decltype(in_end)>(in_end);
    auto out    = OutIter(output);

    func(in, sent, out, N);
    assert(output[0] == make<To>(2));
  }
}

template <class InIter, template <class> class SentWrapper, class OutIter>
constexpr void test_copy() {
  // Classic.
  if constexpr (std::same_as<InIter, SentWrapper<InIter>>) {
    test_one<InIter, SentWrapper, OutIter>([](auto first, auto last, auto out, size_t) {
      std::copy(first, last, out);
    });
    test_one<InIter, SentWrapper, OutIter>([](auto first, auto last, auto out, size_t n) {
      std::copy_backward(first, last, out + n);
    });
    test_one<InIter, SentWrapper, OutIter>([](auto first, auto, auto out, size_t n) {
      std::copy_n(first, n, out);
    });
  }

  // Ranges.
  test_one<InIter, SentWrapper, OutIter>([](auto first, auto last, auto out, size_t) {
    std::ranges::copy(first, last, out);
  });
  test_one<InIter, SentWrapper, OutIter>([](auto first, auto last, auto out, size_t n) {
    std::ranges::copy_backward(first, last, out + n);
  });
  test_one<InIter, SentWrapper, OutIter>([](auto first, auto, auto out, size_t n) {
    std::ranges::copy_n(first, n, out);
  });
}

template <class InIter, template <class> class SentWrapper, class OutIter>
constexpr void test_move() {
  if constexpr (std::same_as<InIter, SentWrapper<InIter>>) {
    test_one<InIter, SentWrapper, OutIter>([](auto first, auto last, auto out, size_t) {
      std::move(first, last, out);
    });
    test_one<InIter, SentWrapper, OutIter>([](auto first, auto last, auto out, size_t n) {
      std::move_backward(first, last, out + n);
    });
  }

  // Ranges.
  test_one<InIter, SentWrapper, OutIter>([](auto first, auto last, auto out, size_t) {
    std::ranges::move(first, last, out);
  });
  test_one<InIter, SentWrapper, OutIter>([](auto first, auto last, auto out, size_t n) {
    std::ranges::move_backward(first, last, out + n);
  });
}

template <class From, class To = From>
constexpr void test_copy_with_type() {
  using FromIter = contiguous_iterator<From*>;
  using ToIter = contiguous_iterator<To*>;

  test_copy<FromIter, std::type_identity_t, ToIter>();
  test_copy<FromIter, sized_sentinel, ToIter>();
  test_copy<FromIter, std::type_identity_t, To*>();
  test_copy<From*, std::type_identity_t, To*>();
  test_copy<From*, std::type_identity_t, ToIter>();
}

template <class From, class To = From>
constexpr void test_move_with_type() {
  using FromIter = contiguous_iterator<From*>;
  using ToIter = contiguous_iterator<To*>;

  test_move<FromIter, std::type_identity_t, ToIter>();
  test_move<FromIter, sized_sentinel, ToIter>();
  test_move<FromIter, std::type_identity_t, To*>();
  test_move<From*, std::type_identity_t, To*>();
  test_move<From*, std::type_identity_t, ToIter>();
}

template <class From, class To>
constexpr void test_copy_and_move() {
  test_copy_with_type<From, To>();
  test_move_with_type<From, To>();
}

template <class From, class To>
constexpr void test_both_directions() {
  test_copy_and_move<From, To>();
  if (!std::same_as<From, To>) {
    test_copy_and_move<To, From>();
  }
}

constexpr bool test() {
  test_copy_with_type<NonTrivialCopyAssignment>();
  test_move_with_type<NonTrivialMoveAssignment>();

  // Copying from a smaller type into a larger type and vice versa.
  test_both_directions<char, int>();
  test_both_directions<std::int32_t, std::int64_t>();

  // Copying between types with different representations.
  test_both_directions<int, float>();
  // Copying from `bool` to `char` will invoke the optimization, so only check one direction.
  test_copy_and_move<char, bool>();

  // Copying between different structs with the same represenation (there is no way to guarantee the representation is
  // the same).
  test_copy_and_move<S1, S2>();
  // Copying between different unions with the same represenation.
  test_copy_and_move<U1, U2>();

  // Copying from a regular pointer to a void pointer (these are not considered trivially copyable).
  test_copy_and_move<int*, void*>();
  // Copying from a non-const pointer to a const pointer (these are not considered trivially copyable).
  test_copy_and_move<int*, const int*>();

  // `memmove` does not support volatile pointers.
  // (See also https://github.com/llvm/llvm-project/issues/28901).
  if (!std::is_constant_evaluated()) {
    test_both_directions<volatile int, int>();
    test_both_directions<volatile int, volatile int>();
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}

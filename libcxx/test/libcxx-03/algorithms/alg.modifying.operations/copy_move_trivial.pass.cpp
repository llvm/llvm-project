//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// In the modules build, adding another overload of `memmove` doesn't work.
// UNSUPPORTED: clang-modules-build
// GCC complains about "ambiguating" `__builtin_memmove`.
// UNSUPPORTED: gcc

// <algorithm>

// These tests check that `std::copy` and `std::move` (including their variations like `copy_n`) forward to
// `memmove` when possible.

#include <cstddef>

struct Foo {
  int i = 0;

  Foo() = default;
  Foo(int set_i) : i(set_i) {}

  friend bool operator==(const Foo&, const Foo&) = default;
};

static bool memmove_called = false;

// This template is a better match than the actual `builtin_memmove` (it can match the pointer type exactly, without an
// implicit conversion to `void*`), so it should hijack the call inside `std::copy` and similar algorithms if it's made.
template <class Dst, class Src>
constexpr void* __builtin_memmove(Dst* dst, Src* src, std::size_t count) {
  memmove_called = true;
  return __builtin_memmove(static_cast<void*>(dst), static_cast<const void*>(src), count);
}

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iterator>
#include <limits>
#include <ranges>
#include <type_traits>

#include "test_iterators.h"

static_assert(std::is_trivially_copyable_v<Foo>);

// To test pointers to functions.
void Func() {}
using FuncPtr = decltype(&Func);

// To test pointers to members.
struct S {
  int mem_obj = 0;
  void MemFunc() {}
};
using MemObjPtr = decltype(&S::mem_obj);
using MemFuncPtr = decltype(&S::MemFunc);

// To test bitfields.
struct BitfieldS {
  unsigned char b1 : 3;
  unsigned char : 2;
  unsigned char b2 : 5;
  friend bool operator==(const BitfieldS&, const BitfieldS&) = default;
};

// To test non-default alignment.
struct AlignedS {
  alignas(64) int x;
  alignas(8) int y;
  friend bool operator==(const AlignedS&, const AlignedS&) = default;
};

template <class T>
T make(int from) {
  return T(from);
}

template <class T>
requires (std::is_pointer_v<T> && !std::is_function_v<std::remove_pointer_t<T>>)
T make(int i) {
  static std::remove_pointer_t<T> arr[8];
  return arr + i;
}

template <class T>
requires std::same_as<T, FuncPtr>
FuncPtr make(int) {
  return &Func;
}

template <class T>
requires std::same_as<T, MemObjPtr>
MemObjPtr make(int) {
  return &S::mem_obj;
}

template <class T>
requires std::same_as<T, MemFuncPtr>
MemFuncPtr make(int) {
  return &S::MemFunc;
}

template <class T>
requires std::same_as<T, BitfieldS>
BitfieldS make(int x) {
  BitfieldS result = {};
  result.b1 = x;
  result.b2 = x;
  return result;
}

template <class T>
requires std::same_as<T, AlignedS>
AlignedS make(int x) {
  AlignedS result;
  result.x = x;
  result.y = x;
  return result;
}

template <class InIter, template <class> class SentWrapper, class OutIter, class Func>
void test_one(Func func) {
  using From = std::iter_value_t<InIter>;
  using To = std::iter_value_t<OutIter>;

  // Normal case.
  {
    const std::size_t N = 4;

    From input[N] = {make<From>(1), make<From>(2), make<From>(3), make<From>(4)};
    To output[N];

    auto in     = InIter(input);
    auto in_end = InIter(input + N);
    auto sent   = SentWrapper<decltype(in_end)>(in_end);
    auto out    = OutIter(output);

    assert(!memmove_called);
    func(in, sent, out, N);
    assert(memmove_called);
    memmove_called = false;

    assert(std::equal(input, input + N, output, [](const From& lhs, const To& rhs) {
        // Prevents warnings/errors due to mismatched signed-ness.
        if constexpr (std::convertible_to<From, To>) {
          return static_cast<To>(lhs) == rhs;
        } else if constexpr (std::convertible_to<To, From>) {
          return lhs == static_cast<From>(rhs);
        }
    }));
  }
}

template <class InIter, template <class> class SentWrapper, class OutIter>
void test_copy_and_move() {
  // Classic.
  if constexpr (std::same_as<InIter, SentWrapper<InIter>>) {
    test_one<InIter, SentWrapper, OutIter>([](auto first, auto last, auto out, std::size_t) {
      std::copy(first, last, out);
    });
    test_one<InIter, SentWrapper, OutIter>([](auto first, auto last, auto out, std::size_t n) {
      std::copy_backward(first, last, out + n);
    });
    test_one<InIter, SentWrapper, OutIter>([](auto first, auto, auto out, std::size_t n) {
      std::copy_n(first, n, out);
    });
    test_one<InIter, SentWrapper, OutIter>([](auto first, auto last, auto out, std::size_t) {
      std::move(first, last, out);
    });
    test_one<InIter, SentWrapper, OutIter>([](auto first, auto last, auto out, std::size_t n) {
      std::move_backward(first, last, out + n);
    });
  }

  // Ranges.
  test_one<InIter, SentWrapper, OutIter>([](auto first, auto last, auto out, std::size_t) {
    std::ranges::copy(first, last, out);
  });
  test_one<InIter, SentWrapper, OutIter>([](auto first, auto last, auto out, std::size_t n) {
    std::ranges::copy_backward(first, last, out + n);
  });
  test_one<InIter, SentWrapper, OutIter>([](auto first, auto, auto out, std::size_t n) {
    std::ranges::copy_n(first, n, out);
  });
  test_one<InIter, SentWrapper, OutIter>([](auto first, auto last, auto out, std::size_t) {
    std::ranges::move(first, last, out);
  });
  test_one<InIter, SentWrapper, OutIter>([](auto first, auto last, auto out, std::size_t n) {
    std::ranges::move_backward(first, last, out + n);
  });
}

template <class From, class To, template <class> class SentWrapper, bool BothDirections = !std::same_as<From, To>>
void test_all_permutations_from_to_sent() {
  test_copy_and_move<From*, SentWrapper, To*>();
  test_copy_and_move<contiguous_iterator<From*>, SentWrapper, To*>();
  test_copy_and_move<From*, SentWrapper, contiguous_iterator<To*>>();
  test_copy_and_move<contiguous_iterator<From*>, SentWrapper, contiguous_iterator<To*>>();

  if (BothDirections) {
    test_copy_and_move<To*, SentWrapper, From*>();
    test_copy_and_move<contiguous_iterator<To*>, SentWrapper, From*>();
    test_copy_and_move<To*, SentWrapper, contiguous_iterator<From*>>();
    test_copy_and_move<contiguous_iterator<To*>, SentWrapper, contiguous_iterator<From*>>();
  }
}

void test_different_signedness() {
  auto check = [](auto alg) {
    // Signed -> unsigned.
    {
      constexpr int N = 3;
      constexpr auto min_value = std::numeric_limits<int>::min();

      int in[N] = {-1, min_value / 2, min_value};
      unsigned int out[N];
      unsigned int expected[N] = {
        static_cast<unsigned int>(in[0]),
        static_cast<unsigned int>(in[1]),
        static_cast<unsigned int>(in[2]),
      };

      assert(!memmove_called);
      alg(in, in + N, out, N);
      assert(memmove_called);
      memmove_called = false;

      assert(std::equal(out, out + N, expected));
    }

    // Unsigned -> signed.
    {
      constexpr int N = 3;
      constexpr auto max_signed = std::numeric_limits<int>::max();
      constexpr auto max_unsigned = std::numeric_limits<unsigned int>::max();

      unsigned int in[N] = {static_cast<unsigned int>(max_signed) + 1, max_unsigned / 2, max_unsigned};
      int out[N];
      int expected[N] = {
        static_cast<int>(in[0]),
        static_cast<int>(in[1]),
        static_cast<int>(in[2]),
      };

      assert(!memmove_called);
      alg(in, in + N, out, N);
      assert(memmove_called);
      memmove_called = false;

      assert(std::equal(out, out + N, expected));
    }
  };

  check([](auto first, auto last, auto out, std::size_t) {
    std::copy(first, last, out);
  });
  check([](auto first, auto last, auto out, std::size_t n) {
    std::copy_backward(first, last, out + n);
  });
  check([](auto first, auto, auto out, std::size_t n) {
    std::copy_n(first, n, out);
  });
  check([](auto first, auto last, auto out, std::size_t) {
    std::move(first, last, out);
  });
  check([](auto first, auto last, auto out, std::size_t n) {
    std::move_backward(first, last, out + n);
  });

  // Ranges.
  check([](auto first, auto last, auto out, std::size_t) {
    std::ranges::copy(first, last, out);
  });
  check([](auto first, auto last, auto out, std::size_t n) {
    std::ranges::copy_backward(first, last, out + n);
  });
  check([](auto first, auto, auto out, std::size_t n) {
    std::ranges::copy_n(first, n, out);
  });
  check([](auto first, auto last, auto out, std::size_t) {
    std::ranges::move(first, last, out);
  });
  check([](auto first, auto last, auto out, std::size_t n) {
    std::ranges::move_backward(first, last, out + n);
  });
}

void test() {
  // Built-in.
  test_all_permutations_from_to_sent<int, int, std::type_identity_t>();
  // User-defined.
  test_all_permutations_from_to_sent<Foo, Foo, std::type_identity_t>();

  // Conversions.
  test_all_permutations_from_to_sent<char32_t, std::int32_t, sized_sentinel>();
  test_all_permutations_from_to_sent<std::int32_t, std::uint32_t, sized_sentinel>();
  // Conversion from `bool` to `char` invokes the optimization (the set of values for `char` is a superset of the set of
  // values for `bool`), but the other way round cannot.
  test_all_permutations_from_to_sent<bool, char, sized_sentinel, /*BothDirections=*/false>();

  // Copying between regular pointers.
  test_copy_and_move<int**, std::type_identity_t, int**>();

  // Copying between pointers to functions.
  test_copy_and_move<FuncPtr*, std::type_identity_t, FuncPtr*>();

  // Copying between pointers to members.
  test_copy_and_move<MemObjPtr*, std::type_identity_t, MemObjPtr*>();
  test_copy_and_move<MemFuncPtr*, std::type_identity_t, MemFuncPtr*>();

  // Copying structs with bitfields.
  test_copy_and_move<BitfieldS*, std::type_identity_t, BitfieldS*>();

  // Copying objects with non-default alignment.
  test_copy_and_move<AlignedS*, std::type_identity_t, AlignedS*>();

  // Copying integers with different signedness produces the same results as built-in assignment.
  test_different_signedness();
}

int main(int, char**) {
  test();
  // The test relies on a global variable, so it cannot be made `constexpr`; the `memmove` optimization is not used in
  // `constexpr` mode anyway.

  return 0;
}

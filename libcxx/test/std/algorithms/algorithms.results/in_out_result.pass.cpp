//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts, libcpp-has-no-incomplete-ranges

// <algorithm>
//
// namespace ranges {
//   template<class InputIterator, class OutputIterator>
//     struct in_out_result;
// }

#include <algorithm>
#include <cassert>
#include <type_traits>

struct A {
  A(int&);
};
static_assert(!std::is_constructible_v<std::ranges::in_out_result<A, A>, std::ranges::in_out_result<int, int>&>);

static_assert(std::is_convertible_v<std::ranges::in_out_result<int, int>&,
    std::ranges::in_out_result<long, long>>);
static_assert(!std::is_nothrow_convertible_v<std::ranges::in_out_result<int, int>&,
    std::ranges::in_out_result<long, long>>);
static_assert(std::is_convertible_v<const std::ranges::in_out_result<int, int>&,
    std::ranges::in_out_result<long, long>>);
static_assert(!std::is_nothrow_convertible_v<const std::ranges::in_out_result<int, int>&,
    std::ranges::in_out_result<long, long>>);
static_assert(std::is_convertible_v<std::ranges::in_out_result<int, int>&&,
    std::ranges::in_out_result<long, long>>);
static_assert(!std::is_nothrow_convertible_v<std::ranges::in_out_result<int, int>&&,
    std::ranges::in_out_result<long, long>>);
static_assert(std::is_convertible_v<const std::ranges::in_out_result<int, int>&&,
    std::ranges::in_out_result<long, long>>);
static_assert(!std::is_nothrow_convertible_v<const std::ranges::in_out_result<int, int>&&,
    std::ranges::in_out_result<long, long>>);

int main(int, char**) {
  // Conversion, fundamental types.
  {
    std::ranges::in_out_result<int, bool> x = {2, false};
    // FIXME(varconst): try a narrowing conversion.
    std::ranges::in_out_result<long, char> y = x;
    assert(y.in == 2);
    assert(y.out == '\0');
  }

  // Conversion, user-defined types.
  {
    struct From1 {
      int value = 0;
      From1(int v) : value(v) {}
    };

    struct To1 {
      int value = 0;
      To1(int v) : value(v) {}

      To1(const From1& f) : value(f.value) {};
    };

    struct To2 {
      int value = 0;
      To2(int v) : value(v) {}
    };
    struct From2 {
      int value = 0;
      From2(int v) : value(v) {}

      operator To2() const { return To2(value); }
    };

    std::ranges::in_out_result<From1, From2> x{42, 99};
    std::ranges::in_out_result<To1, To2> y = x;
    assert(y.in.value == 42);
    assert(y.out.value == 99);
  }

  // Copy-only type.
  {
    struct CopyOnly {
      int value = 0;
      CopyOnly() = default;
      CopyOnly(int v) : value(v) {}

      CopyOnly(const CopyOnly&) = default;
      CopyOnly(CopyOnly&&) = delete;
    };

    std::ranges::in_out_result<CopyOnly, CopyOnly> x;
    x.in.value = 42;
    x.out.value = 99;

    auto y = x;
    assert(y.in.value == 42);
    assert(y.out.value == 99);
  }

  // Move-only type.
  {
    struct MoveOnly {
      int value = 0;
      MoveOnly(int v) : value(v) {}

      MoveOnly(MoveOnly&&) = default;
      MoveOnly(const MoveOnly&) = delete;
    };

    std::ranges::in_out_result<MoveOnly, MoveOnly> x{42, 99};
    auto y = std::move(x);
    assert(y.in.value == 42);
    assert(y.out.value == 99);
  }

  // Unsuccessful conversion.
  {
    struct Foo1 {};
    struct Foo2 {};
    struct Bar1 {};
    struct Bar2 {};
    static_assert(
        !std::is_convertible_v<std::ranges::in_out_result<Foo1, Foo2>, std::ranges::in_out_result<Bar1, Bar2>>);
  }

  return 0;
}

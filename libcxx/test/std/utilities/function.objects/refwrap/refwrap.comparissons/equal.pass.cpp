//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20, c++23

// <functional>

// class reference_wrapper

// // [refwrap.comparisons], comparisons
// friend constexpr bool operator==(reference_wrapper, reference_wrapper);                                // Since C++26
// friend constexpr bool operator==(reference_wrapper, const T&);                                         // Since C++26
// friend constexpr bool operator==(reference_wrapper, reference_wrapper<const T>);                       // Since C++26

#include <cassert>
#include <concepts>
#include <functional>

#include "test_comparisons.h"
#include "test_macros.h"

struct EqualityComparable {
  constexpr EqualityComparable(int value) : value_{value} {};

  friend constexpr bool operator==(const EqualityComparable&, const EqualityComparable&) noexcept = default;

  int value_;
};

static_assert(std::equality_comparable<EqualityComparable>);
static_assert(EqualityComparable{94} == EqualityComparable{94});
static_assert(EqualityComparable{94} != EqualityComparable{82});

struct NonComparable {};

static_assert(!std::equality_comparable<NonComparable>);

// Test SFINAE.

static_assert(std::equality_comparable<std::reference_wrapper<EqualityComparable>>);

static_assert(!std::equality_comparable<std::reference_wrapper<NonComparable>>);

// // refwrap, refwrap
// static_assert(AssertEqualityAreNoexcept<std::reference_wrapper<EqualityComparable>>());
// static_assert(AssertEqualityReturnBool<std::reference_wrapper<EqualityComparable>>());
// // refwrap, const&
// static_assert(AssertEqualityAreNoexcept<std::reference_wrapper<EqualityComparable>, int>());
// static_assert(AssertEqualityReturnBool<std::reference_wrapper<EqualityComparable>, int>());
// // refwrap, refwrap<const>
// static_assert(AssertEqualityAreNoexcept<std::reference_wrapper<EqualityComparable>,
//                                         std::reference_wrapper<const EqualityComparable>>());
// static_assert(AssertEqualityReturnBool<std::reference_wrapper<EqualityComparable>,
//                                        std::reference_wrapper<const EqualityComparable>>());

// Test equality.

template <typename T>
constexpr void test() {
  T i{92};
  T j{84};

  // `operator==`
  {
    // refwrap, refwrap
    std::reference_wrapper<T> rw1{i};
    std::reference_wrapper<T> rw2 = rw1;
    // std::reference_wrapper<T> rw3{j};
    // static_assert(AssertEqualityAreNoexcept<decltype(rw1), decltype(rw1)>);
    // static_assert(AssertEqualityReturnBool<decltype(rw1), decltype(rw1)>);
    std::same_as<bool> decltype(auto) _ = rw1 == rw1;
    assert(rw1 == rw1);
    // static_assert(AssertEqualityAreNoexcept<decltype(rw1), decltype(rw2)>);
    // static_assert(AssertEqualityReturnBool<decltype(rw1), decltype(rw2)>);
    std::same_as<bool> decltype(auto) _ = rw1 == rw2;
    assert(rw1 == rw2);
    // static_assert(AssertEqualityAreNoexcept<decltype(rw2), decltype(rw1)>);
    // static_assert(AssertEqualityReturnBool<decltype(rw2), decltype(rw1)>);
    std::same_as<bool> decltype(auto) _ = rw2 == rw1;
    assert(rw2 == rw1);
  }
  {
    // refwrap, const&
    std::reference_wrapper<T> rw{i};
    // static_assert(AssertEqualityAreNoexcept<decltype(rw), decltype(i)>);
    // static_assert(AssertEqualityReturnBool<decltype(rw), decltype(i)>);
    assert(rw == i);
    // static_assert(AssertEqualityAreNoexcept<decltype(i), decltype(rw)>);
    // static_assert(AssertEqualityReturnBool<decltype(i), decltype(rw)>);
    assert(i == rw);
  }
  {
    // refwrap, refwrap<const>
    std::reference_wrapper<T> rw1{i};
    std::reference_wrapper<const T> rw2 = rw1;
    // static_assert(AssertEqualityAreNoexcept<decltype(rw1), decltype(rw2)>);
    // static_assert(AssertEqualityReturnBool<decltype(rw1), decltype(rw2)>);
    assert(rw1 == rw2);
    // static_assert(AssertEqualityAreNoexcept<decltype(rw2), decltype(rw1)>);
    // static_assert(AssertEqualityReturnBool<decltype(rw2), decltype(rw1)>);
    assert(rw2 == rw1);
  }

  // `operator!=`
  {
    // refwrap, refwrap
    std::reference_wrapper<T> rw1{i};
    std::reference_wrapper<T> rw2{j};
    assert(rw1 != rw2);
    assert(rw2 != rw1);
  }
  {
    // refwrap, const&
    std::reference_wrapper<T> rw{i};
    assert(rw != j);
    assert(j != rw);
  }
  {
    // refwrap, refwrap<const>
    std::reference_wrapper<T> rw1{i};
    std::reference_wrapper<const T> rw2{j};
    assert(rw1 != rw2);
    assert(rw2 != rw1);
  }
}

constexpr bool test() {
  test<int>();
  test<EqualityComparable>();

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}

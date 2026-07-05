//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <tuple>

// template <class... Types> class tuple;

// template<class... TTypes, class... UTypes>
//   bool
//   operator==(const tuple<TTypes...>& t, const tuple<UTypes...>& u);
// template<tuple-like UTuple>
//   friend constexpr bool operator==(const tuple& t, const UTuple& u); // since C++23

// UNSUPPORTED: c++03

#include <array>
#include <cassert>
#include <tuple>

#include "test_comparisons.h"
#include "test_macros.h"

#if TEST_STD_VER >= 23
#  include <ranges>
#endif
#if TEST_STD_VER >= 26
#  include <complex>
#endif

#if TEST_STD_VER >= 26

// Test SFINAE.

static_assert(std::equality_comparable<std::tuple<EqualityComparable>>);
static_assert(std::equality_comparable<std::tuple<EqualityComparable, EqualityComparable>>);

static_assert(!std::equality_comparable<std::tuple<NonComparable>>);
static_assert(!std::equality_comparable<std::tuple<EqualityComparable, NonComparable>>);
static_assert(!std::equality_comparable<std::tuple<NonComparable, EqualityComparable>>);
static_assert(!std::equality_comparable_with<std::tuple<EqualityComparable>, std::tuple<NonComparable>>);
static_assert(!std::equality_comparable_with<std::tuple<NonComparable>, std::tuple<EqualityComparable>>);
// Size mismatch.
static_assert(
    !std::equality_comparable_with<std::tuple<EqualityComparable>, std::tuple<EqualityComparable, EqualityComparable>>);
static_assert(
    !std::equality_comparable_with<std::tuple<EqualityComparable, EqualityComparable>, std::tuple<EqualityComparable>>);

// Heterogeneous comparisons.
// TODO: Use equality_comparable_with once other changes of tuple introduced in P2165R4 are implemented.
template <class T, class U>
concept can_eq_compare = requires(const T& t, const U& u) { t == u; };

static_assert(can_eq_compare<std::tuple<EqualityComparable>, std::array<EqualityComparable, 1>>);
static_assert(!can_eq_compare<std::tuple<EqualityComparable>, std::array<NonComparable, 1>>);

static_assert(can_eq_compare<std::tuple<EqualityComparable, EqualityComparable>,
                             std::pair<EqualityComparable, EqualityComparable>>);
static_assert(
    !can_eq_compare<std::tuple<EqualityComparable, EqualityComparable>, std::pair<EqualityComparable, NonComparable>>);

static_assert(can_eq_compare<std::tuple<int*, int*>, std::ranges::subrange<const int*>>);
static_assert(!can_eq_compare<std::tuple<int (*)[1], int (*)[1]>, std::ranges::subrange<const int*>>);
static_assert(can_eq_compare<std::tuple<double, double>, std::complex<float>>);
static_assert(!can_eq_compare<std::tuple<int*, int*>, std::complex<float>>);

// Size mismatch in heterogeneous comparisons.
static_assert(!can_eq_compare<std::tuple<>, std::array<EqualityComparable, 2>>);
static_assert(!can_eq_compare<std::tuple<EqualityComparable>, std::array<EqualityComparable, 2>>);
static_assert(!can_eq_compare<std::tuple<>, std::pair<EqualityComparable, EqualityComparable>>);
static_assert(!can_eq_compare<std::tuple<EqualityComparable>, std::pair<EqualityComparable, EqualityComparable>>);
static_assert(!can_eq_compare<std::tuple<int*>, std::ranges::subrange<int*>>);
static_assert(!can_eq_compare<std::tuple<double>, std::complex<double>>);

#endif

TEST_CONSTEXPR_CXX14 bool test() {
  {
    typedef std::tuple<> T1;
    typedef std::tuple<> T2;
    const T1 t1;
    const T2 t2;
    assert(t1 == t2);
    assert(!(t1 != t2));
  }
  {
    typedef std::tuple<int> T1;
    typedef std::tuple<double> T2;
    const T1 t1(1);
    const T2 t2(1.1);
    assert(!(t1 == t2));
    assert(t1 != t2);
  }
  {
    typedef std::tuple<int> T1;
    typedef std::tuple<double> T2;
    const T1 t1(1);
    const T2 t2(1);
    assert(t1 == t2);
    assert(!(t1 != t2));
  }
  {
    typedef std::tuple<int, double> T1;
    typedef std::tuple<double, long> T2;
    const T1 t1(1, 2);
    const T2 t2(1, 2);
    assert(t1 == t2);
    assert(!(t1 != t2));
  }
  {
    typedef std::tuple<int, double> T1;
    typedef std::tuple<double, long> T2;
    const T1 t1(1, 2);
    const T2 t2(1, 3);
    assert(!(t1 == t2));
    assert(t1 != t2);
  }
  {
    typedef std::tuple<int, double> T1;
    typedef std::tuple<double, long> T2;
    const T1 t1(1, 2);
    const T2 t2(1.1, 2);
    assert(!(t1 == t2));
    assert(t1 != t2);
  }
  {
    typedef std::tuple<int, double> T1;
    typedef std::tuple<double, long> T2;
    const T1 t1(1, 2);
    const T2 t2(1.1, 3);
    assert(!(t1 == t2));
    assert(t1 != t2);
  }
  {
    typedef std::tuple<long, int, double> T1;
    typedef std::tuple<double, long, int> T2;
    const T1 t1(1, 2, 3);
    const T2 t2(1, 2, 3);
    assert(t1 == t2);
    assert(!(t1 != t2));
  }
  {
    typedef std::tuple<long, int, double> T1;
    typedef std::tuple<double, long, int> T2;
    const T1 t1(1, 2, 3);
    const T2 t2(1.1, 2, 3);
    assert(!(t1 == t2));
    assert(t1 != t2);
  }
  {
    typedef std::tuple<long, int, double> T1;
    typedef std::tuple<double, long, int> T2;
    const T1 t1(1, 2, 3);
    const T2 t2(1, 3, 3);
    assert(!(t1 == t2));
    assert(t1 != t2);
  }
  {
    typedef std::tuple<long, int, double> T1;
    typedef std::tuple<double, long, int> T2;
    const T1 t1(1, 2, 3);
    const T2 t2(1, 2, 4);
    assert(!(t1 == t2));
    assert(t1 != t2);
  }
  {
    typedef std::tuple<long, int, double> T1;
    typedef std::tuple<double, long, int> T2;
    const T1 t1(1, 2, 3);
    const T2 t2(1, 3, 2);
    assert(!(t1 == t2));
    assert(t1 != t2);
  }
  {
    typedef std::tuple<long, int, double> T1;
    typedef std::tuple<double, long, int> T2;
    const T1 t1(1, 2, 3);
    const T2 t2(1.1, 2, 2);
    assert(!(t1 == t2));
    assert(t1 != t2);
  }
  {
    typedef std::tuple<long, int, double> T1;
    typedef std::tuple<double, long, int> T2;
    const T1 t1(1, 2, 3);
    const T2 t2(1.1, 3, 3);
    assert(!(t1 == t2));
    assert(t1 != t2);
  }
  {
    typedef std::tuple<long, int, double> T1;
    typedef std::tuple<double, long, int> T2;
    const T1 t1(1, 2, 3);
    const T2 t2(1.1, 3, 2);
    assert(!(t1 == t2));
    assert(t1 != t2);
  }
#if TEST_STD_VER >= 14
  {
    using T1 = std::tuple<long, int, double>;
    using T2 = std::tuple<double, long, int>;
    constexpr T1 t1(1, 2, 3);
    constexpr T2 t2(1.1, 3, 2);
    assert(!(t1 == t2));
    assert(t1 != t2);
  }
#endif
#if TEST_STD_VER >= 23
  {
    using T1 = std::tuple<long, int>;
    using T2 = std::pair<double, long>;
    constexpr T1 t1{1, 2};
    constexpr T2 t2{1.0, 2};
    assert(t1 == t2);
    assert(!(t1 != t2));
  }
  {
    using T1 = std::tuple<long, int>;
    using T2 = std::pair<double, long>;
    constexpr T1 t1{1, 2};
    constexpr T2 t2{1.1, 3};
    assert(!(t1 == t2));
    assert(t1 != t2);
  }
  {
    using T1 = std::tuple<long, int>;
    using T2 = std::array<double, 2>;
    constexpr T1 t1{1, 2};
    constexpr T2 t2{1.0, 2.0};
    assert(t1 == t2);
    assert(!(t1 != t2));
  }
  {
    using T1 = std::tuple<long, int>;
    using T2 = std::array<double, 2>;
    constexpr T1 t1{1, 2};
    constexpr T2 t2{1.1, 3.0};
    assert(!(t1 == t2));
    assert(t1 != t2);
  }
  {
    using T1 = std::tuple<const int*, const int*>;
    using T2 = std::ranges::subrange<const int*>;

    int arr[1]{};
    T1 t1{arr, arr + 1};
    T2 t2{arr};
    assert(t1 == t2);
    assert(!(t1 != t2));
  }
  {
    using T1 = std::tuple<const int*, const int*>;
    using T2 = std::ranges::subrange<const int*>;

    int arr[1]{};
    T1 t1{arr, arr};
    T2 t2{arr};
    assert(!(t1 == t2));
    assert(t1 != t2);
  }
  {
    assert((std::tuple<>{} == std::array<int*, 0>{}));
    assert((std::tuple<>{} == std::array<double, 0>{}));
  }
#endif
#if TEST_STD_VER >= 26
  {
    using T1 = std::tuple<long, int>;
    using T2 = std::complex<double>;
    constexpr T1 t1{1, 2};
    constexpr T2 t2{1.0, 2.0};
    assert(t1 == t2);
    assert(!(t1 != t2));
  }
  {
    using T1 = std::tuple<long, int>;
    using T2 = std::complex<double>;
    constexpr T1 t1{1, 2};
    constexpr T2 t2{1.1, 3.0};
    assert(!(t1 == t2));
    assert(t1 != t2);
  }
#endif

  return true;
}

int main(int, char**) {
  test();
#if TEST_STD_VER >= 14
  static_assert(test(), "");
#endif
  return 0;
}

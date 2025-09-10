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
//   operator<(const tuple<TTypes...>& t, const tuple<UTypes...>& u);
//
// template<class... TTypes, class... UTypes>
//   bool
//   operator>(const tuple<TTypes...>& t, const tuple<UTypes...>& u);
//
// template<class... TTypes, class... UTypes>
//   bool
//   operator<=(const tuple<TTypes...>& t, const tuple<UTypes...>& u);
//
// template<class... TTypes, class... UTypes>
//   bool
//   operator>=(const tuple<TTypes...>& t, const tuple<UTypes...>& u);

// UNSUPPORTED: c++03

#include <cassert>
#include <tuple>

#include "test_macros.h"

#if TEST_STD_VER >= 23
#  include <array>
#  include <ranges>
#  include <utility>
#endif
#if TEST_STD_VER >= 26
#  include <complex>
#endif

TEST_CONSTEXPR_CXX14 bool test() {
  {
    typedef std::tuple<> T1;
    typedef std::tuple<> T2;
    const T1 t1;
    const T2 t2;
    assert(!(t1 < t2));
    assert((t1 <= t2));
    assert(!(t1 > t2));
    assert((t1 >= t2));
  }
  {
    typedef std::tuple<long> T1;
    typedef std::tuple<double> T2;
    const T1 t1(1);
    const T2 t2(1);
    assert(!(t1 < t2));
    assert((t1 <= t2));
    assert(!(t1 > t2));
    assert((t1 >= t2));
  }
  {
    typedef std::tuple<long> T1;
    typedef std::tuple<double> T2;
    const T1 t1(1);
    const T2 t2(0.9);
    assert(!(t1 < t2));
    assert(!(t1 <= t2));
    assert((t1 > t2));
    assert((t1 >= t2));
  }
  {
    typedef std::tuple<long> T1;
    typedef std::tuple<double> T2;
    const T1 t1(1);
    const T2 t2(1.1);
    assert((t1 < t2));
    assert((t1 <= t2));
    assert(!(t1 > t2));
    assert(!(t1 >= t2));
  }
  {
    typedef std::tuple<long, int> T1;
    typedef std::tuple<double, long> T2;
    const T1 t1(1, 2);
    const T2 t2(1, 2);
    assert(!(t1 < t2));
    assert((t1 <= t2));
    assert(!(t1 > t2));
    assert((t1 >= t2));
  }
  {
    typedef std::tuple<long, int> T1;
    typedef std::tuple<double, long> T2;
    const T1 t1(1, 2);
    const T2 t2(0.9, 2);
    assert(!(t1 < t2));
    assert(!(t1 <= t2));
    assert((t1 > t2));
    assert((t1 >= t2));
  }
  {
    typedef std::tuple<long, int> T1;
    typedef std::tuple<double, long> T2;
    const T1 t1(1, 2);
    const T2 t2(1.1, 2);
    assert((t1 < t2));
    assert((t1 <= t2));
    assert(!(t1 > t2));
    assert(!(t1 >= t2));
  }
  {
    typedef std::tuple<long, int> T1;
    typedef std::tuple<double, long> T2;
    const T1 t1(1, 2);
    const T2 t2(1, 1);
    assert(!(t1 < t2));
    assert(!(t1 <= t2));
    assert((t1 > t2));
    assert((t1 >= t2));
  }
  {
    typedef std::tuple<long, int> T1;
    typedef std::tuple<double, long> T2;
    const T1 t1(1, 2);
    const T2 t2(1, 3);
    assert((t1 < t2));
    assert((t1 <= t2));
    assert(!(t1 > t2));
    assert(!(t1 >= t2));
  }
  {
    typedef std::tuple<long, int, double> T1;
    typedef std::tuple<double, long, int> T2;
    const T1 t1(1, 2, 3);
    const T2 t2(1, 2, 3);
    assert(!(t1 < t2));
    assert((t1 <= t2));
    assert(!(t1 > t2));
    assert((t1 >= t2));
  }
  {
    typedef std::tuple<long, int, double> T1;
    typedef std::tuple<double, long, int> T2;
    const T1 t1(1, 2, 3);
    const T2 t2(0.9, 2, 3);
    assert(!(t1 < t2));
    assert(!(t1 <= t2));
    assert((t1 > t2));
    assert((t1 >= t2));
  }
  {
    typedef std::tuple<long, int, double> T1;
    typedef std::tuple<double, long, int> T2;
    const T1 t1(1, 2, 3);
    const T2 t2(1.1, 2, 3);
    assert((t1 < t2));
    assert((t1 <= t2));
    assert(!(t1 > t2));
    assert(!(t1 >= t2));
  }
  {
    typedef std::tuple<long, int, double> T1;
    typedef std::tuple<double, long, int> T2;
    const T1 t1(1, 2, 3);
    const T2 t2(1, 1, 3);
    assert(!(t1 < t2));
    assert(!(t1 <= t2));
    assert((t1 > t2));
    assert((t1 >= t2));
  }
  {
    typedef std::tuple<long, int, double> T1;
    typedef std::tuple<double, long, int> T2;
    const T1 t1(1, 2, 3);
    const T2 t2(1, 3, 3);
    assert((t1 < t2));
    assert((t1 <= t2));
    assert(!(t1 > t2));
    assert(!(t1 >= t2));
  }
  {
    typedef std::tuple<long, int, double> T1;
    typedef std::tuple<double, long, int> T2;
    const T1 t1(1, 2, 3);
    const T2 t2(1, 2, 2);
    assert(!(t1 < t2));
    assert(!(t1 <= t2));
    assert((t1 > t2));
    assert((t1 >= t2));
  }
  {
    typedef std::tuple<long, int, double> T1;
    typedef std::tuple<double, long, int> T2;
    const T1 t1(1, 2, 3);
    const T2 t2(1, 2, 4);
    assert((t1 < t2));
    assert((t1 <= t2));
    assert(!(t1 > t2));
    assert(!(t1 >= t2));
  }
#if TEST_STD_VER >= 14
  {
    using T1 = std::tuple<long, int, double>;
    using T2 = std::tuple<double, long, int>;
    constexpr T1 t1(1, 2, 3);
    constexpr T2 t2(1, 2, 4);
    assert((t1 < t2));
    assert((t1 <= t2));
    assert(!(t1 > t2));
    assert(!(t1 >= t2));
  }
#endif
#if TEST_STD_VER >= 23
  {
    using T1 = std::tuple<long, int>;
    using T2 = std::pair<double, long>;
    constexpr T1 t1{1, 2};
    constexpr T2 t2{1.0, 2};
    assert(!(t1 < t2));
    assert(t1 <= t2);
    assert(!(t1 > t2));
    assert(t1 >= t2);
  }
  {
    using T1 = std::tuple<long, int>;
    using T2 = std::pair<double, long>;
    constexpr T1 t1{1, 2};
    constexpr T2 t2{1.0, 3};
    assert(t1 < t2);
    assert(t1 <= t2);
    assert(!(t1 > t2));
    assert(!(t1 >= t2));
  }
  {
    using T1 = std::tuple<long, int>;
    using T2 = std::array<double, 2>;
    constexpr T1 t1{1, 2};
    constexpr T2 t2{1.0, 2.0};
    assert(!(t1 < t2));
    assert(t1 <= t2);
    assert(!(t1 > t2));
    assert(t1 >= t2);
  }
  {
    using T1 = std::tuple<long, int>;
    using T2 = std::array<double, 2>;
    constexpr T1 t1{1, 2};
    constexpr T2 t2{1.1, 3.0};
    assert(t1 < t2);
    assert(t1 <= t2);
    assert(!(t1 > t2));
    assert(!(t1 >= t2));
  }
  {
    using T1 = std::tuple<const int*, const int*>;
    using T2 = std::ranges::subrange<const int*>;

    int arr[1]{};
    T1 t1{arr, arr + 1};
    T2 t2{arr};
    assert(!(t1 < t2));
    assert(t1 <= t2);
    assert(!(t1 > t2));
    assert(t1 >= t2);
  }
  {
    using T1 = std::tuple<const int*, const int*>;
    using T2 = std::ranges::subrange<const int*>;

    int arr[1]{};
    T1 t1{arr + 1, arr + 1};
    T2 t2{arr};
    assert(!(t1 < t2));
    assert(!(t1 <= t2));
    assert(t1 > t2);
    assert(t1 >= t2);
  }
  {
    constexpr std::tuple<> t1{};
    constexpr std::array<int*, 0> t2{};
    assert(!(t1 < t2));
    assert(t1 <= t2);
    assert(!(t1 > t2));
    assert(t1 >= t2);
  }
  {
    constexpr std::tuple<> t1{};
    constexpr std::array<double, 0> t2{};
    assert(!(t1 < t2));
    assert(t1 <= t2);
    assert(!(t1 > t2));
    assert(t1 >= t2);
  }
#endif
#if TEST_STD_VER >= 26
  {
    using T1 = std::tuple<long, int>;
    using T2 = std::complex<double>;
    constexpr T1 t1{1, 2};
    constexpr T2 t2{1.0, 2.0};
    assert(!(t1 < t2));
    assert(t1 <= t2);
    assert(!(t1 > t2));
    assert(t1 >= t2);
  }
  {
    using T1 = std::tuple<long, int>;
    using T2 = std::complex<double>;
    constexpr T1 t1{1, 2};
    constexpr T2 t2{1.1, 3.0};
    assert(t1 < t2);
    assert(t1 <= t2);
    assert(!(t1 > t2));
    assert(!(t1 >= t2));
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

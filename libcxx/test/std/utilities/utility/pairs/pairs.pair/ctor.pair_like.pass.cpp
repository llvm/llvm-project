//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <utility>

// template <class T1, class T2> struct pair

// template <pair-like P>
// constexpr explicit(see-below) pair(P&&); // since C++23

#include <array>
#include <cassert>
#include <complex>
#include <ranges>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>

#include "test_macros.h"

namespace my_ns{

struct MyPairLike {

template <std::size_t N>
friend int get(MyPairLike const&)
{
  return 0;
}

};

} // namespace my_ns

template <>
struct std::tuple_size<my_ns::MyPairLike> : std::integral_constant<std::size_t, 2> {};

template <std::size_t N>
struct std::tuple_element<N, my_ns::MyPairLike> {
  using type = int;
};

// https://llvm.org/PR65620
// This used to be a hard error
static_assert(!std::is_constructible_v<std::pair<int,int>, my_ns::MyPairLike const&>);


constexpr bool test() {
  // Make sure construction works from array, tuple, and ranges::subrange
  {
    // Check from std::array
    {
      std::array<int, 2> a = {1, 2};
      std::pair<int, int> p(a);
      assert(p.first == 1);
      assert(p.second == 2);
      static_assert(!std::is_constructible_v<std::pair<int, int>, std::array<int, 1>>); // too small
      static_assert( std::is_constructible_v<std::pair<int, int>, std::array<int, 2>>); // works (test the test)
      static_assert(!std::is_constructible_v<std::pair<int, int>, std::array<int, 3>>); // too large
    }

    // Check from std::tuple
    {
      std::tuple<int, int> a = {1, 2};
      std::pair<int, int> p(a);
      assert(p.first == 1);
      assert(p.second == 2);
      static_assert(!std::is_constructible_v<std::pair<int, int>, std::tuple<int>>); // too small
      static_assert( std::is_constructible_v<std::pair<int, int>, std::tuple<int, int>>); // works (test the test)
      static_assert(!std::is_constructible_v<std::pair<int, int>, std::tuple<int, int, int>>); // too large
    }

    // Check that the constructor excludes ranges::subrange
    {
      int data[] = {1, 2, 3, 4, 5};
      const std::ranges::subrange a(data);
      // Note the expression below would be ambiguous if pair's
      // constructor does not exclude subrange
      std::pair<int*, int*> p = a;
      assert(p.first == data + 0);
      assert(p.second == data + 5);
    }
  }

  // Make sure we allow element conversion from a pair-like
  {
    std::tuple<int, char const*> a = {34, "hello world"};
    std::pair<long, std::string> p(a);
    assert(p.first == 34);
    assert(p.second == std::string("hello world"));
    static_assert(!std::is_constructible_v<std::pair<long, std::string>, std::tuple<char*, std::string>>); // first not convertible
    static_assert(!std::is_constructible_v<std::pair<long, std::string>, std::tuple<long, void*>>); // second not convertible
    static_assert( std::is_constructible_v<std::pair<long, std::string>, std::tuple<long, std::string>>); // works (test the test)
  }

  // Make sure we forward the pair-like elements
  {
    struct NoCopy {
      NoCopy() = default;
      NoCopy(NoCopy const&) = delete;
      NoCopy(NoCopy&&) = default;
    };
    std::tuple<NoCopy, NoCopy> a;
    std::pair<NoCopy, NoCopy> p(std::move(a));
    (void)p;
  }

  // Make sure the constructor is implicit iff both elements can be converted
  {
    struct To { };
    struct FromImplicit {
      constexpr operator To() const { return To{}; }
    };
    struct FromExplicit {
      constexpr explicit operator To() const { return To{}; }
    };
    // If both are convertible, the constructor is not explicit
    {
      std::tuple<FromImplicit, float> a = {FromImplicit{}, 2.3f};
      std::pair<To, double> p = a;
      (void)p;
      static_assert(std::is_convertible_v<std::tuple<FromImplicit, float>, std::pair<To, double>>);
    }
    // Otherwise, the constructor is explicit
    {
      static_assert( std::is_constructible_v<std::pair<To, int>, std::tuple<FromExplicit, int>>);
      static_assert(!std::is_convertible_v<std::tuple<FromExplicit, int>, std::pair<To, int>>);

      static_assert( std::is_constructible_v<std::pair<int, To>, std::tuple<int, FromExplicit>>);
      static_assert(!std::is_convertible_v<std::tuple<int, FromExplicit>, std::pair<int, To>>);

      static_assert( std::is_constructible_v<std::pair<To, To>, std::tuple<FromExplicit, FromExplicit>>);
      static_assert(!std::is_convertible_v<std::tuple<FromExplicit, FromExplicit>, std::pair<To, To>>);
    }
  }

  // Test construction prohibition of introduced by https://wg21.link/P2255R2.

  // Test tuple.
  {
    static_assert(!std::is_constructible_v<std::pair<const int&, const long&>, std::tuple<char, long>>);
    static_assert(!std::is_constructible_v<std::pair<const int&, const long&>, const std::tuple<char, long>&>);
    static_assert(!std::is_convertible_v<std::tuple<char, long>, std::pair<const int&, const long&>>);
    static_assert(!std::is_convertible_v<const std::tuple<char, long>&, std::pair<const int&, const long&>>);

    static_assert(!std::is_constructible_v<std::pair<const int&, const long&>, std::tuple<int, short>>);
    static_assert(!std::is_constructible_v<std::pair<const int&, const long&>, const std::tuple<int, short>&>);
    static_assert(!std::is_convertible_v<std::tuple<int, short>, std::pair<const int&, const long&>>);
    static_assert(!std::is_convertible_v<const std::tuple<char, short>&, std::pair<const int&, const long&>>);
  }
  // Test array.
  {
    static_assert(!std::is_constructible_v<std::pair<const int&, const long&>, std::array<int, 2>>);
    static_assert(!std::is_constructible_v<std::pair<const long&, const int&>, std::array<int, 2>>);
    static_assert(!std::is_constructible_v<std::pair<const int&, const long&>, const std::array<int, 2>&>);
    static_assert(!std::is_constructible_v<std::pair<const long&, const int&>, const std::array<int, 2>&>);

    static_assert(!std::is_convertible_v<std::array<int, 2>, std::pair<const int&, const long&>>);
    static_assert(!std::is_convertible_v<std::array<int, 2>, std::pair<const long&, const int&>>);
    static_assert(!std::is_convertible_v<const std::array<int, 2>&, std::pair<const int&, const long&>>);
    static_assert(!std::is_convertible_v<const std::array<int, 2>&, std::pair<const long&, const int&>>);
  }
#if TEST_STD_VER >= 26
  // Test complex.
  {
    static_assert(!std::is_constructible_v<std::pair<const double&, const float&>, std::complex<float>>);
    static_assert(!std::is_constructible_v<std::pair<const float&, const double&>, std::complex<float>>);
    static_assert(!std::is_convertible_v<std::complex<float>, std::pair<const double&, const float&>>);
    static_assert(!std::is_convertible_v<std::complex<float>, std::pair<const float&, const double&>>);

    static_assert(!std::is_constructible_v<std::pair<const double&, const long double&>, const std::complex<double>&>);
    static_assert(!std::is_constructible_v<std::pair<const long double&, const double&>, const std::complex<double>&>);
    static_assert(!std::is_convertible_v<const std::complex<double>&, std::pair<const double&, const long double&>>);
    static_assert(!std::is_convertible_v<const std::complex<double>&, std::pair<const long double&, const double&>>);
  }
#endif // TEST_STD_VER >= 26

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}

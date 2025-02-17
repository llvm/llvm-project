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
#include <ranges>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>

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

// https://github.com/llvm/llvm-project/issues/65620
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
  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}

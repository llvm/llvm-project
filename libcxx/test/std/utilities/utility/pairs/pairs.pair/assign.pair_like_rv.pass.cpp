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

//  template <pair-like P> constexpr pair& operator=(P&&);  // since C++23

#include <array>
#include <cassert>
#include <concepts>
#include <ranges>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>

constexpr bool test() {
  // Make sure assignment works from array and tuple
  {
    // Check from std::array
    {
      std::array<int, 2> a = {1, 2};
      std::pair<int, int> p;
      std::same_as<std::pair<int, int>&> decltype(auto) result = (p = a);
      assert(&result == &p);
      assert(p.first == 1);
      assert(p.second == 2);
      static_assert(!std::is_assignable_v<std::pair<int, int>&, std::array<int, 1>>); // too small
      static_assert( std::is_assignable_v<std::pair<int, int>&, std::array<int, 2>>); // works (test the test)
      static_assert(!std::is_assignable_v<std::pair<int, int>&, std::array<int, 3>>); // too large
    }

    // Check from std::tuple
    {
      std::tuple<int, int> a = {1, 2};
      std::pair<int, int> p;
      std::same_as<std::pair<int, int>&> decltype(auto) result = (p = a);
      assert(&result == &p);
      assert(p.first == 1);
      assert(p.second == 2);
      static_assert(!std::is_assignable_v<std::pair<int, int>&, std::tuple<int>>); // too small
      static_assert( std::is_assignable_v<std::pair<int, int>&, std::tuple<int, int>>); // works (test the test)
      static_assert(!std::is_assignable_v<std::pair<int, int>&, std::tuple<int, int, int>>); // too large
    }

    // Make sure it works for ranges::subrange. This actually deserves an explanation: even though
    // the assignment operator explicitly excludes ranges::subrange specializations, such assignments
    // end up working because of ranges::subrange's implicit conversion to pair-like types.
    // This test ensures that the interoperability works as intended.
    {
      struct Assignable {
        int* ptr = nullptr;
        Assignable() = default;
        constexpr Assignable(int* p) : ptr(p) { } // enable `subrange::operator pair-like`
        constexpr Assignable& operator=(Assignable const&) = default;
      };
      int data[] = {1, 2, 3, 4, 5};
      std::ranges::subrange<int*> a(data);
      std::pair<Assignable, Assignable> p;
      std::same_as<std::pair<Assignable, Assignable>&> decltype(auto) result = (p = a);
      assert(&result == &p);
      assert(p.first.ptr == data);
      assert(p.second.ptr == data + 5);
    }
  }

  // Make sure we allow element conversion from a pair-like
  {
    std::tuple<int, char const*> a = {34, "hello world"};
    std::pair<long, std::string> p;
    std::same_as<std::pair<long, std::string>&> decltype(auto) result = (p = a);
    assert(&result == &p);
    assert(p.first == 34);
    assert(p.second == std::string("hello world"));
    static_assert(!std::is_assignable_v<std::pair<long, std::string>&, std::tuple<char*, std::string>>); // first not convertible
    static_assert(!std::is_assignable_v<std::pair<long, std::string>&, std::tuple<long, void*>>); // second not convertible
    static_assert( std::is_assignable_v<std::pair<long, std::string>&, std::tuple<long, std::string>>); // works (test the test)
  }

  // Make sure we forward the pair-like elements
  {
    struct NoCopy {
      NoCopy() = default;
      NoCopy(NoCopy const&) = delete;
      NoCopy(NoCopy&&) = default;
      NoCopy& operator=(NoCopy const&) = delete;
      NoCopy& operator=(NoCopy&&) = default;
    };
    std::tuple<NoCopy, NoCopy> a;
    std::pair<NoCopy, NoCopy> p;
    p = std::move(a);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}

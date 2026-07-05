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

//  template <pair-like P> constexpr const pair& operator=(P&&) const;  // since C++23

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
      int x = 91, y = 92;
      std::array<int, 2> a = {1, 2};
      std::pair<int&, int&> const p = {x, y};
      std::same_as<std::pair<int&, int&> const&> decltype(auto) result = (p = a);
      assert(&result == &p);
      assert(x == 1);
      assert(y == 2);
      static_assert(!std::is_assignable_v<std::pair<int&, int&> const&, std::array<int, 1>>); // too small
      static_assert( std::is_assignable_v<std::pair<int&, int&> const&, std::array<int, 2>>); // works (test the test)
      static_assert(!std::is_assignable_v<std::pair<int&, int&> const&, std::array<int, 3>>); // too large
    }

    // Check from std::tuple
    {
      int x = 91, y = 92;
      std::tuple<int, int> a = {1, 2};
      std::pair<int&, int&> const p = {x, y};
      std::same_as<std::pair<int&, int&> const&> decltype(auto) result = (p = a);
      assert(&result == &p);
      assert(x == 1);
      assert(y == 2);
      static_assert(!std::is_assignable_v<std::pair<int&, int&> const&, std::tuple<int>>); // too small
      static_assert( std::is_assignable_v<std::pair<int&, int&> const&, std::tuple<int, int>>); // works (test the test)
      static_assert(!std::is_assignable_v<std::pair<int&, int&> const&, std::tuple<int, int, int>>); // too large
    }

    // Make sure it works for ranges::subrange. This actually deserves an explanation: even though
    // the assignment operator explicitly excludes ranges::subrange specializations, such assignments
    // end up working because of ranges::subrange's implicit conversion to pair-like types.
    // This test ensures that the interoperability works as intended.
    {
      struct ConstAssignable {
        mutable int* ptr = nullptr;
        ConstAssignable() = default;
        constexpr ConstAssignable(int* p) : ptr(p) { } // enable `subrange::operator pair-like`
        constexpr ConstAssignable const& operator=(ConstAssignable const& other) const { ptr = other.ptr; return *this; }

        constexpr ConstAssignable(ConstAssignable const&) = default; // defeat -Wdeprecated-copy
        constexpr ConstAssignable& operator=(ConstAssignable const&) = default; // defeat -Wdeprecated-copy
      };
      int data[] = {1, 2, 3, 4, 5};
      std::ranges::subrange<int*> a(data);
      std::pair<ConstAssignable, ConstAssignable> const p;
      std::same_as<std::pair<ConstAssignable, ConstAssignable> const&> decltype(auto) result = (p = a);
      assert(&result == &p);
      assert(p.first.ptr == data);
      assert(p.second.ptr == data + 5);
    }
  }

  // Make sure we allow element conversion from a pair-like
  {
    struct ConstAssignable {
      mutable int val = 0;
      ConstAssignable() = default;
      constexpr ConstAssignable const& operator=(int v) const { val = v; return *this; }
    };
    std::tuple<int, int> a = {1, 2};
    std::pair<ConstAssignable, ConstAssignable> const p;
    std::same_as<std::pair<ConstAssignable, ConstAssignable> const&> decltype(auto) result = (p = a);
    assert(&result == &p);
    assert(p.first.val == 1);
    assert(p.second.val == 2);
    static_assert(!std::is_assignable_v<std::pair<ConstAssignable, ConstAssignable> const&, std::tuple<void*, int>>); // first not convertible
    static_assert(!std::is_assignable_v<std::pair<ConstAssignable, ConstAssignable> const&, std::tuple<int, void*>>); // second not convertible
    static_assert( std::is_assignable_v<std::pair<ConstAssignable, ConstAssignable> const&, std::tuple<int, int>>); // works (test the test)
  }

  // Make sure we forward the pair-like elements
  {
    struct NoCopy {
      NoCopy() = default;
      NoCopy(NoCopy const&) = delete;
      NoCopy(NoCopy&&) = default;
      NoCopy& operator=(NoCopy const&) = delete;
      constexpr NoCopy const& operator=(NoCopy&&) const { return *this; }
    };
    std::tuple<NoCopy, NoCopy> a;
    std::pair<NoCopy, NoCopy> const p;
    p = std::move(a);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}

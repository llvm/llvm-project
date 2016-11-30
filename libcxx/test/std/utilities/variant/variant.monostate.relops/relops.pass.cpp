// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11, c++14

// <variant>

// constexpr bool operator<(monostate, monostate) noexcept { return false; }
// constexpr bool operator>(monostate, monostate) noexcept { return false; }
// constexpr bool operator<=(monostate, monostate) noexcept { return true; }
// constexpr bool operator>=(monostate, monostate) noexcept { return true; }
// constexpr bool operator==(monostate, monostate) noexcept { return true; }
// constexpr bool operator!=(monostate, monostate) noexcept { return false; }

#include <cassert>
#include <type_traits>
#include <variant>

int main() {
  using M = std::monostate;
  constexpr M m1{};
  constexpr M m2{};
  {
    static_assert((m1 < m2) == false, "");
    static_assert(noexcept(m1 < m2), "");
  }
  {
    static_assert((m1 > m2) == false, "");
    static_assert(noexcept(m1 > m2), "");
  }
  {
    static_assert((m1 <= m2) == true, "");
    static_assert(noexcept(m1 <= m2), "");
  }
  {
    static_assert((m1 >= m2) == true, "");
    static_assert(noexcept(m1 >= m2), "");
  }
  {
    static_assert((m1 == m2) == true, "");
    static_assert(noexcept(m1 == m2), "");
  }
  {
    static_assert((m1 != m2) == false, "");
    static_assert(noexcept(m1 != m2), "");
  }
}

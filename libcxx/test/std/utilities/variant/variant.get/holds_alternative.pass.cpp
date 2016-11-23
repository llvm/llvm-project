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

// template <class T, class... Types>
// constexpr bool holds_alternative(const variant<Types...>& v) noexcept;

#include <variant>

int main() {
  {
    using V = std::variant<int>;
    constexpr V v;
    static_assert(std::holds_alternative<int>(v), "");
  }
  {
    using V = std::variant<int, long>;
    constexpr V v;
    static_assert(std::holds_alternative<int>(v), "");
    static_assert(!std::holds_alternative<long>(v), "");
  }
  { // noexcept test
    using V = std::variant<int>;
    const V v;
    static_assert(noexcept(std::holds_alternative<int>(v)), "must be noexcept");
  }
}

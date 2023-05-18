//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// <variant>

//  template <class T, class... Types>
//  constexpr add_pointer_t<T> get_if(variant<Types...>* v) noexcept;
// template <class T, class... Types>
//  constexpr add_pointer_t<const T> get_if(const variant<Types...>* v)
//  noexcept;

#include "test_macros.h"
#include "variant_test_helpers.h"
#include <cassert>
#include <variant>

void test_const_get_if() {
  {
    using V = std::variant<int>;
    constexpr const V *v = nullptr;
    static_assert(std::get_if<int>(v) == nullptr, "");
  }
  {
    using V = std::variant<int, const long>;
    constexpr V v(42);
    ASSERT_NOEXCEPT(std::get_if<int>(&v));
    ASSERT_SAME_TYPE(decltype(std::get_if<int>(&v)), const int *);
    static_assert(*std::get_if<int>(&v) == 42, "");
    static_assert(std::get_if<const long>(&v) == nullptr, "");
  }
  {
    using V = std::variant<int, const long>;
    constexpr V v(42l);
    ASSERT_SAME_TYPE(decltype(std::get_if<const long>(&v)), const long *);
    static_assert(*std::get_if<const long>(&v) == 42, "");
    static_assert(std::get_if<int>(&v) == nullptr, "");
  }
}

void test_get_if() {
  {
    using V = std::variant<int>;
    V *v = nullptr;
    assert(std::get_if<int>(v) == nullptr);
  }
  {
    using V = std::variant<int, const long>;
    V v(42);
    ASSERT_NOEXCEPT(std::get_if<int>(&v));
    ASSERT_SAME_TYPE(decltype(std::get_if<int>(&v)), int *);
    assert(*std::get_if<int>(&v) == 42);
    assert(std::get_if<const long>(&v) == nullptr);
  }
  {
    using V = std::variant<int, const long>;
    V v(42l);
    ASSERT_SAME_TYPE(decltype(std::get_if<const long>(&v)), const long *);
    assert(*std::get_if<const long>(&v) == 42);
    assert(std::get_if<int>(&v) == nullptr);
  }
}

int main(int, char**) {
  test_const_get_if();
  test_get_if();

  return 0;
}

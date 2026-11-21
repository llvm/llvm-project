//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26
// ADDITIONAL_COMPILE_FLAGS: -Wno-unused-value

// constant_wrapper

// template<constexpr-param L, constexpr-param R>
//   friend constexpr auto operator->*(L, R) noexcept -> constant_wrapper<L::value->*(R::value)>
//     { return {}; }

#include <cassert>
#include <concepts>
#include <utility>

#include "helpers.h"

struct S {
  int member = 42;
};

constexpr S s;

template <class L, class R>
concept HasPtrToMem = requires(L l, R r) {
  { l->*r };
};

template <class L, class R>
concept HasNoexceptPtrToMem = requires(L l, R r) {
  { l->*r } noexcept;
};

struct WithOps {
  int value;
  constexpr WithOps(int v) : value(v) {}

  friend constexpr auto operator->*(WithOps w, int WithOps::* pm) { return w.value + (&w)->*pm; }
};

struct OpsReturnNonStructural {
  int value;
  constexpr OpsReturnNonStructural(int v) : value(v) {}

  friend constexpr auto operator->*(OpsReturnNonStructural o, int OpsReturnNonStructural::* pm) {
    return NonStructural{o.value + (&o)->*pm};
  }
};

struct NoOps {};

static_assert(HasPtrToMem<std::constant_wrapper<&s>, std::constant_wrapper<&S::member>>);
static_assert(HasNoexceptPtrToMem<std::constant_wrapper<&s>, std::constant_wrapper<&S::member>>);

static_assert(HasPtrToMem<std::constant_wrapper<&s>, int S::*>);
static_assert(!HasPtrToMem<std::constant_wrapper<&s>, int>);

constexpr bool test() {
  {
    // use builtin operator->*
    std::constant_wrapper<(&s)> cwS;
    std::constant_wrapper<&S::member> cwPM;
    std::same_as<std::constant_wrapper<42>> decltype(auto) result1 = cwS->*cwPM;
    static_assert(result1 == 42);
  }

  {
    // mix runtime and constant_wrapper parameters, will use built-in operator
    std::constant_wrapper<(&s)> cwS;
    int S::* pm                                     = &S::member;
    std::same_as<const int&> decltype(auto) result1 = cwS->*pm;
    assert(result1 == 42);
  }

  {
    // custom operator->*
    std::constant_wrapper<WithOps{42}> cwWO;
    std::constant_wrapper<&WithOps::value> cwPM;
    std::same_as<std::constant_wrapper<84>> decltype(auto) result1 = cwWO->*cwPM;
    static_assert(result1 == 84);
  }

  {
    // Return non-structural type
    // Will use underlying type's runtime operators
    std::constant_wrapper<OpsReturnNonStructural{42}> cwORNS;
    std::constant_wrapper<&OpsReturnNonStructural::value> cwPM;
    std::same_as<NonStructural> decltype(auto) result1 = cwORNS->*cwPM;
    assert(result1.get() == 84);
  }

  {
    // integral_constant
    std::constant_wrapper<(&s)> cwS;
    std::integral_constant<int S::*, &S::member> icPM;
    std::same_as<std::constant_wrapper<42>> decltype(auto) result1 = cwS->*icPM;
    static_assert(result1 == 42);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}

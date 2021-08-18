//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// template <class F1, class F2>
// constexpr unspecified __compose(F1&&, F2&&);

#include <functional>
#include <cassert>
#include <tuple>
#include <utility>

template <int X>
struct Elem {
  template <int Y>
  constexpr bool operator==(Elem<Y> const&) const
  { return X == Y; }
};

struct F {
  template <class ...Args>
  constexpr auto operator()(Args&& ...args) const {
    return std::make_tuple(Elem<888>{}, std::forward<Args>(args)...);
  }
};

struct G {
  template <class ...Args>
  constexpr auto operator()(Args&& ...args) const {
    return std::make_tuple(Elem<999>{}, std::forward<Args>(args)...);
  }
};

constexpr bool test() {
  F const f;
  G const g;

  {
    auto c = std::__compose(f, g);
    assert(c() == f(g()));
  }
  {
    auto c = std::__compose(f, g);
    assert(c(Elem<0>{}) == f(g(Elem<0>{})));
  }
  {
    auto c = std::__compose(f, g);
    assert(c(Elem<0>{}, Elem<1>{}) == f(g(Elem<0>{}, Elem<1>{})));
  }
  {
    auto c = std::__compose(f, g);
    assert(c(Elem<0>{}, Elem<1>{}, Elem<2>{}) == f(g(Elem<0>{}, Elem<1>{}, Elem<2>{})));
  }

  // Make sure we can call a function that's a pointer to a member function.
  {
    struct MemberFunction1 {
      constexpr Elem<0> foo() { return {}; }
    };
    struct MemberFunction2 {
      constexpr MemberFunction1 bar() { return {}; }
    };
    auto c = std::__compose(&MemberFunction1::foo, &MemberFunction2::bar);
    assert(c(MemberFunction2{}) == Elem<0>{});
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}

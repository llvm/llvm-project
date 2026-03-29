//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Test the set of C++11 features that Clang provides as an extension in C++03 mode.
// The language features we expect are:
//
// 1. rvalue references (and perfect forwarding)
// 2. variadic templates
// 3. alias templates
// 4. defaulted and deleted functions.
// 5. default values for non-type template parameters.
//
// Some features we don't get and can't be used in extended C++03 mode:
//
// 1. noexcept and constexpr
// 2. Two closing '>' without a space.

#include <type_traits>
#include <cassert>

// Equals delete and default are allowed in minimal C++03 mode.
namespace test_eq_delete_and_default {
void t1() = delete;
struct T2 {
  T2() = default;
  T2(T2 const&) = delete;
};
}

namespace alias_templates {
template <class T>
using X = T;
static_assert((std::is_same<X<int>, int>::value), "");
}

namespace variadics_templates {
template <class ...Args>
int t1(Args...) {
  return sizeof...(Args);
}
void test() {
  assert(t1() == 0);
  assert(t1(42) == 1);
  assert(t1(1, 2, 3) == 3);
}
}

namespace rvalue_references_move_semantics {
struct T {
  T() : moved(0) {}
  T(T const& other) : moved(other.moved) {}
  T(T&& other) : moved(other.moved) { ++moved; other.moved = -1; }
  int moved;
};
void f(T o, int expect_moved) { assert(o.moved == expect_moved); }
void test() {
  {
    T t;
    assert(t.moved == 0);
    T t2(static_cast<T&&>(t));
    assert(t2.moved == 1);
    assert(t.moved == -1);
  }
  {
    T t;
    f(t, 0);
    f(static_cast<T&&>(t), 1);
  }
}
}

namespace rvalue_references_perfect_forwarding {
template <class Expect, class T>
void f(T&&) {
  static_assert((std::is_same<Expect, T&&>::value), "");
}
void test() {
  int x = 42;
  f<int&>(x);
  f<int&&>(42);
  f<int&&>(static_cast<int&&>(x));
}
}

namespace default_values_for_nttp {
template <int I = 42>
void f() { assert(I == 42); }
void test() {
  f();
}
}

namespace reference_qualified_functions {
struct T {
  T() : lvalue_called(0), rvalue_called(0) {}
  void foo() const & { lvalue_called++; }
  void foo() && { rvalue_called++; }
  mutable int lvalue_called;
  int rvalue_called;
};

void test() {
  {
    T t;
    t.foo();
    assert(t.lvalue_called == 1);
    assert(t.rvalue_called == 0);
  }
  {
    T t;
    static_cast<T&&>(t).foo();
    assert(t.lvalue_called == 0);
    assert(t.rvalue_called == 1);
  }
}
}

int main(int, char**) {
  variadics_templates::test();
  rvalue_references_move_semantics::test();
  rvalue_references_perfect_forwarding::test();
  default_values_for_nttp::test();
  reference_qualified_functions::test();
  return 0;
}

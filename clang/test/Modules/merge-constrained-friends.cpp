// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++23 %t/A.cppm -emit-module-interface -o %t/A.pcm
// RUN: %clang_cc1 -std=c++23 %t/Use.cpp -fprebuilt-module-path=%t -fsyntax-only -verify

//--- A.cppm
module;
export module A;

struct B {};

export template<int N> struct A : B {
  friend constexpr const int *f(B) requires true {
    static constexpr int result = N;
    return &result;
  }

  template<int M>
  friend constexpr const int *g(B) requires (M >= 0) && (N >= 0) {
    static constexpr int result = M * 10 + N;
    return &result;
  }
};

export inline A<1> a1;
export inline A<2> a2;
export inline A<3> a3;

static_assert(f(a1) != f(a2) && f(a2) != f(a3));
static_assert(g<1>(a1) != g<1>(a2) && g<1>(a2) != g<1>(a3));

static_assert(*f(a1) == 1);
static_assert(*f(a2) == 2);
static_assert(*f(a3) == 3);

static_assert(*g<4>(a1) == 41);
static_assert(*g<5>(a2) == 52);
static_assert(*g<6>(a3) == 63);

//--- Use.cpp
// expected-no-diagnostics
import A;

// Try some instantiations we tried before and some we didn't.
static_assert(f(a1) != f(a2) && f(a2) != f(a3));
static_assert(g<1>(a1) != g<1>(a2) && g<1>(a2) != g<1>(a3));
static_assert(g<2>(a1) != g<2>(a2) && g<2>(a2) != g<2>(a3));

A<4> a4;
static_assert(f(a1) != f(a4) && f(a2) != f(a4) && f(a3) != f(a4));
static_assert(g<3>(a1) != g<3>(a4));

static_assert(*f(a1) == 1);
static_assert(*f(a2) == 2);
static_assert(*f(a3) == 3);
static_assert(*f(a4) == 4);

static_assert(*g<4>(a1) == 41);
static_assert(*g<5>(a2) == 52);
static_assert(*g<6>(a3) == 63);

static_assert(*g<7>(a1) == 71);
static_assert(*g<8>(a4) == 84);

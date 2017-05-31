// RUN: %clang_cc1 -pedantic -std=c++1y %s -o %t
// RUN: %clang_cc1 -pedantic -std=c++1y -emit-pch %s -o %t
// RUN: %clang_cc1 -pedantic -std=c++1y -include-pch %t -verify %s

#ifndef HEADER_INCLUDED

#define HEADER_INCLUDED

struct A {
  int x;
  int y = 3;
  int z = x + y;
};
template<typename T> constexpr A make() { return A {}; }
template<typename T> constexpr A make(T t) { return A { t }; }

struct B {
  int z1, z2 = z1;
  constexpr B(int k) : z1(k) {}
};

#else

static_assert(A{}.z == 3, "");
static_assert(A{1}.z == 4, "");
static_assert(A{.y = 5}.z == 5, ""); // expected-warning {{C99}}
static_assert(A{3, .y = 1}.z == 4, ""); // expected-warning {{C99}}
static_assert(make<int>().z == 3, "");
static_assert(make<int>(12).z == 15, "");

#endif

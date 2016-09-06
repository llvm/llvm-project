// RUN: %clang_cc1 -pedantic -std=c++1y -include %s -include %s -verify %s
// RUN: %clang_cc1 -pedantic -std=c++1y -emit-pch -o %t.1 %s
// RUN: %clang_cc1 -pedantic -std=c++1y -include-pch %t.1 -emit-pch -o %t.2 %s
// RUN: %clang_cc1 -pedantic -std=c++1y -include-pch %t.2 -verify %s

#ifndef HEADER_1
#define HEADER_1

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

template<typename T> struct C {
  constexpr C() {}
  T c = T();
  struct U {};
};
// Instantiate C<int> but not the default initializer.
C<int>::U ciu;

#elif !defined(HEADER_2)
#define HEADER_2

// Instantiate the default initializer now, should create an update record.
C<int> ci;

#else

static_assert(A{}.z == 3, "");
static_assert(A{1}.z == 4, "");
static_assert(A{.y = 5}.z == 5, ""); // expected-warning {{C99}}
static_assert(A{3, .y = 1}.z == 4, ""); // expected-warning {{C99}}
static_assert(make<int>().z == 3, "");
static_assert(make<int>(12).z == 15, "");
static_assert(C<int>().c == 0, "");

#endif

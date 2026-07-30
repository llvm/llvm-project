// RUN: %clang_cc1 -pedantic-errors -std=c++11 -emit-pch %s -o %t.1
// RUN: %clang_cc1 -pedantic-errors -std=c++11 -include-pch %t.1 -emit-pch %s -o %t.2
// RUN: %clang_cc1 -pedantic-errors -std=c++11 -include-pch %t.2 -verify %s
// RUN: %clang_cc1 -pedantic-errors -std=c++11 -include-pch %t.2 -emit-llvm-only %s

// RUN: %clang_cc1 -pedantic-errors -std=c++11 -emit-pch -fpch-instantiate-templates %s -o %t.1
// RUN: %clang_cc1 -pedantic-errors -std=c++11 -include-pch %t.1 -emit-pch -fpch-instantiate-templates %s -o %t.2
// RUN: %clang_cc1 -pedantic-errors -std=c++11 -include-pch %t.2 -verify %s
// RUN: %clang_cc1 -pedantic-errors -std=c++11 -include-pch %t.2 -emit-llvm-only %s

// expected-no-diagnostics

#ifndef PHASE1_DONE
#define PHASE1_DONE

template<int n> int f() noexcept(n % 2) { return 0; }
template<int n> int g() noexcept(n % 2);

decltype(f<2>()) f0;
decltype(f<3>()) f1;
template int f<4>();
template int f<5>();
decltype(f<6>()) f6;
decltype(f<7>()) f7;

struct A {
  A();
  A(const A&);
};

decltype(g<0>()) g0;

#elif !defined(PHASE2_DONE)
#define PHASE2_DONE

template int f<6>();
template int f<7>();
decltype(f<8>()) f8;
decltype(f<9>()) f9;
template int f<10>();
template int f<11>();

A::A() = default;
A::A(const A&) = default;

int g0val = g<0>();

#else

static_assert(!noexcept(f<0>()), "");
static_assert(noexcept(f<1>()), "");

#endif

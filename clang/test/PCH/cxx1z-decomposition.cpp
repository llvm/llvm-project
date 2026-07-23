// No PCH:
// RUN: %clang_cc1 -pedantic -std=c++1z -include %s -verify %s
// RUN: not %clang_cc1 -pedantic -std=c++1z -fsyntax-only %s 2>&1 | FileCheck %s
//
// With PCH:
// RUN: %clang_cc1 -pedantic -std=c++1z -emit-pch -fallow-pch-with-compiler-errors %s -o %t
// RUN: %clang_cc1 -pedantic -std=c++1z -include-pch %t -fallow-pch-with-compiler-errors -verify %s

// RUN: %clang_cc1 -pedantic -std=c++1z -emit-pch -fallow-pch-with-compiler-errors -fpch-instantiate-templates %s -o %t
// RUN: %clang_cc1 -pedantic -std=c++1z -include-pch %t -fallow-pch-with-compiler-errors -verify %s

#ifndef HEADER
#define HEADER

template<typename T> auto decomp(const T &t) {
  auto &[a, b] = t;
  return a + b;
}

struct Q { int a, b; };
constexpr int foo(Q &&q) {
  auto &[a, b] = q;
  return a * 10 + b;
}

auto [noinit]; // expected-error{{structured binding declaration '[noinit]' requires an initializer; expected '=' or braced initializer list}}
               // CHECK: :[[@LINE-1]]:14: error: structured binding declaration '[noinit]' requires an initializer; expected '=' or braced initializer list

#else

int arr[2];
int k = decomp(arr);

static_assert(foo({1, 2}) == 12);

// expected-error@16 {{cannot bind non-class, non-array type 'const int'}}
int z = decomp(10); // expected-note {{instantiation of}}

#endif

// No PCH:
// RUN: %clang_cc1 -pedantic -std=c++1z -include %s -verify %s
//
// With PCH:
// RUN: %clang_cc1 -pedantic -std=c++1z -emit-pch %s -o %t
// RUN: %clang_cc1 -pedantic -std=c++1z -include-pch %t -verify %s

// RUN: %clang_cc1 -pedantic -std=c++1z -emit-pch -fpch-instantiate-templates %s -o %t
// RUN: %clang_cc1 -pedantic -std=c++1z -include-pch %t -verify %s

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

#else

int arr[2];
int k = decomp(arr);

static_assert(foo({1, 2}) == 12);

// expected-error@15 {{cannot decompose non-class, non-array type 'const int'}}
int z = decomp(10); // expected-note {{instantiation of}}

#endif

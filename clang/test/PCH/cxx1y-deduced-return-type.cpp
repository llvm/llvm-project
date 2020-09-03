// No PCH:
// RUN: %clang_cc1 -pedantic -std=c++1y -include %s -include %s -verify %s
//
// With chained PCH:
// RUN: %clang_cc1 -pedantic -std=c++1y -emit-pch %s -o %t.a
// RUN: %clang_cc1 -pedantic -std=c++1y -include-pch %t.a -emit-pch %s -o %t.b
// RUN: %clang_cc1 -pedantic -std=c++1y -include-pch %t.b -verify %s

// RUN: %clang_cc1 -pedantic -std=c++1y -emit-pch -fpch-instantiate-templates %s -o %t.a
// RUN: %clang_cc1 -pedantic -std=c++1y -include-pch %t.a -emit-pch -fpch-instantiate-templates %s -o %t.b
// RUN: %clang_cc1 -pedantic -std=c++1y -include-pch %t.b -verify %s

// expected-no-diagnostics

#if !defined(HEADER1)
#define HEADER1

auto &f(int &);

template<typename T> decltype(auto) g(T &t) {
  return f(t);
}

#elif !defined(HEADER2)
#define HEADER2

// Ensure that this provides an update record for the type of HEADER1's 'f',
// so that HEADER1's 'g' can successfully call it.
auto &f(int &n) {
  return n;
}

#else

int n;
int &k = g(n);

#endif

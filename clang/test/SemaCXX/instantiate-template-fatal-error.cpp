// RUN: %clang_cc1 -fsyntax-only -verify -std=c++14 %s

#pragma clang diagnostic fatal "-Wall"
#pragma clang diagnostic fatal "-Wold-style-cast"

template <class T> bool foo0(const long long *a, T* b) {
  return a == (const long long*)b; // expected-error {{use of old-style cast}}
}

template<class T>
struct S1 {
};

template<class T>
struct S2 : S1<T> {
  bool m1(const long long *a, T *b) const { return foo0(a, b); }
};

bool foo1(const long long *a, int *b) {
  S2<int> s2;
  return s2.m1(a, b);
}

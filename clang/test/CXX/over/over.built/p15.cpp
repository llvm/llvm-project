// RUN: %clang_cc1 -std=c++11 -verify %s -Wno-tautological-compare

template <typename T, typename U>
void f(int* pi, float* pf, T* pt, U* pu, T t) {
  (void)(pi - pi);
  (void)(pi - pf);  // expected-error {{not pointers to compatible types}}
  (void)(pi - pt);
  (void)(pu - pi);
  (void)(pu - pt);
  (void)(pu - t);
  (void)(pi - t);
}

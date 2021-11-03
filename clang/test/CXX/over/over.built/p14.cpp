// RUN: %clang_cc1 -std=c++11 -verify %s -Wno-tautological-compare

template <typename T>
void f(int* pi, T* pt) {
  (void)(pi+3);
  (void)(3+pi);
  (void)(pi-3);
  (void)(pi[3]);
  (void)(3[pi]);

  (void)(pt+3);
  (void)(3+pt);
  (void)(pt-3);
  (void)(pt[3]);
  (void)(3[pt]);
}
// expected-no-diagnostics

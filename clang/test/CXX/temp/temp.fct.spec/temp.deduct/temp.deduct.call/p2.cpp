// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics
template<typename T> struct A { };

// bullet 1
template<typename T> A<T> f0(T* ptr);

void test_f0_bullet1() {
  int arr0[6];
  A<int> a0 = f0(arr0);
  const int arr1[] = { 1, 2, 3, 4, 5 };
  A<const int> a1 = f0(arr1);
}

// bullet 2
int g0(int, int);
float g1(float);

void test_f0_bullet2() {
  A<int(int, int)> a0 = f0(g0);
  A<float(float)> a1 = f0(g1);
}

// bullet 3
struct X { };
const X get_X();

template<typename T> A<T> f1(T);

void test_f1_bullet3() {
  A<X> a0 = f1(get_X());
}

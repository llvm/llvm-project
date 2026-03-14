// RUN: %clang_cc1 %s -emit-llvm-only -verify
// expected-no-diagnostics

struct F {
  void (*x)();
};
void G();
template<class T> class A {
public: A();
};
template<class T> A<T>::A() {
  static F f = { G };
}
A<int> a;

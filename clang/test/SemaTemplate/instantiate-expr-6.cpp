// RUN: %clang_cc1 -std=c++11 -emit-llvm-only %s

struct X {
  template<typename T> static typename T::type g(T t);
  template<typename T> auto f(T t) -> decltype(g(t));
  void f(...);
};

void test() {
  X().f(0);
  X().f(0);
}

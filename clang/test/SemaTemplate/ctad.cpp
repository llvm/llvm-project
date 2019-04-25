// RUN: %clang_cc1 -std=c++17 -verify %s

// expected-no-diagnostics
namespace pr41427 {
  template <typename T> class A {
  public:
    A(void (*)(T)) {}
  };
  
  void D(int) {}
  
  void f() {
    A a(&D);
    using T = decltype(a);
    using T = A<int>;
  }
}

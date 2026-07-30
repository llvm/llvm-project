// RUN: %clang_cc1 -verify -emit-llvm-only %s

namespace test0 {
  template<typename T> unsigned f0() {
    return T::MaxSize; // expected-error {{'int' cannot be used prior to '::'}}
  };
  template<typename T> struct A {
    void Allocate(unsigned Alignment
                    = f0<T>()) // expected-note {{in instantiation}}
    {}
  };
  void f1(A<int> x) { x.Allocate(); }
  
}

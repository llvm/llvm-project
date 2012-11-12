// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s

void f0() {
  int &ir = { 17 }; // expected-error{{reference to type 'int' cannot bind to an initializer list}}
}

namespace PR12453 {
  template<typename T>
  void f(int i) {
    T x{i}; // expected-error{{non-constant-expression cannot be narrowed from type 'int' to 'float' in initializer list}} \
    // expected-note{{override this message by inserting an explicit cast}}
    T y{i}; // expected-error{{non-constant-expression cannot be narrowed from type 'int' to 'float' in initializer list}} \
    // expected-note{{override this message by inserting an explicit cast}}
  }

  template void f<float>(int); // expected-note{{in instantiation of function template specialization 'PR12453::f<float>' requested here}}
}

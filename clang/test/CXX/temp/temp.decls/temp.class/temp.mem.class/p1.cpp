// RUN: %clang_cc1 -fsyntax-only -verify %s

template<typename T, typename U>
struct X0 {
  struct Inner;
};

template<typename T, typename U>
struct X0<T, U>::Inner {
  T x;
  U y;
  
  void f() { x = y; } // expected-error{{incompatible}}
};


void test(int i, float f) {
  X0<int, float>::Inner inner;
  inner.x = 5;
  inner.y = 3.4;
  inner.f();
  
  X0<int*, float *>::Inner inner2;
  inner2.x = &i;
  inner2.y = &f;
  inner2.f(); // expected-note{{instantiation}}
}

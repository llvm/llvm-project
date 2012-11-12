// RUN: %clang_cc1 -fsyntax-only -verify %s
template<typename T> struct A {
  struct B { };
  
  friend struct B;
};

void f() {
  A<int>::B b;
}

struct C0 {
  friend struct A<int>;
};

namespace PR6770 {
  namespace N {
    int f1(int);
  }
  using namespace N;

  namespace M { 
    float f1(float);
  }
  using M::f1;

  template<typename T> void f1(T, T);
  template <class T>
  void f() {
    friend class f; // expected-error{{'friend' used outside of class}}
    friend class f1; // expected-error{{'friend' used outside of class}}
  }
}

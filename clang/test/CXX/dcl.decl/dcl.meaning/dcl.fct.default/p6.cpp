// RUN: %clang_cc1 -fsyntax-only -verify %s

class C { 
public:
  void f(int i = 3); // expected-note{{here}}
  void g(int i, int j = 99);
};

void C::f(int i = 3) { } // expected-error{{redefinition of default argument}}

void C::g(int i = 88, int j) { }

void test_C(C c) {
  c.f();
  c.g();
}

template<typename T>
struct X0 {
  void f(int);
  
  struct Inner {
    void g(int);
  };
};

// DR217
template<typename T>
void X0<T>::f(int = 17) { } // expected-error{{cannot be added}}

// DR217 + DR205 (reading tea leaves)
template<typename T>
void X0<T>::Inner::g(int = 17) { } // expected-error{{cannot be added}}

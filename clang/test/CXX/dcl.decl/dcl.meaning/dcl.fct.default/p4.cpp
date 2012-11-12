// RUN: %clang_cc1 -fsyntax-only -verify %s

void f0(int i, int j, int k = 3);
void f0(int i, int j, int k);
void f0(int i, int j = 2, int k);
void f0(int i, int j, int k);
void f0(int i = 1, // expected-note{{previous definition}}
        int j, int k);
void f0(int i, int j, int k);

namespace N0 {
  void f0(int, int, int); // expected-note{{candidate}}

  void test_f0_inner_scope() {
    f0(); // expected-error{{no matching}}
  }
}

void test_f0_outer_scope() {
  f0(); // okay
}

void f0(int i = 1, // expected-error{{redefinition of default argument}}
        int, int); 

template<typename T> void f1(T); // expected-note{{previous}}

template<typename T>
void f1(T = T()); // expected-error{{cannot be added}}


namespace N1 {
  // example from C++03 standard
  // FIXME: make these "f2"s into "f"s, then fix our scoping issues
  void f2(int, int); 
  void f2(int, int = 7); 
  void h() {
    f2(3); // OK, calls f(3, 7) 
    void f(int = 1, int);	// expected-error{{missing default argument}}
  }
  
  void m()
  {
    void f(int, int); // expected-note{{'f' declared here}}
    f(4);  // expected-error{{too few arguments to function call}}
    void f(int, int = 5); // expected-note{{previous definition}}
    f(4); // okay
    void f(int, int = 5); // expected-error{{redefinition of default argument}}
  }
  
  void n()
  {
    f2(6); // okay
  }
}

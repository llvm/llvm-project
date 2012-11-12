// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics

typedef int f; 

namespace N0 {
  struct A { 
    friend void f(); 
    void g() {
      int i = f(1);
    }
  };
}

namespace N1 {
  struct A { 
    friend void f(A &);
    operator int();
    void g(A a) {
      // ADL should not apply to the lookup of 'f', it refers to the typedef
      // above.
      int i = f(a);
    }
  };
}

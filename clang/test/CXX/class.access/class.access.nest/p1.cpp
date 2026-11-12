// RUN: %clang_cc1 -fsyntax-only -verify %s

// Derived from GNU's std::string
namespace test0 {
  class A {
    struct B {
      unsigned long length;
    };
    struct C : B {
      static const unsigned long max_length;
    };
  };
  
  const unsigned long A::C::max_length = sizeof(B);
}

// Example from the standard.
namespace test1 {
  class E {
    int x;
    class B {};

    class I {
      B b;
      int y; // expected-note {{declared private here}}
      void f(E* p, int i) {
        p->x = i;
      }
    };

    int g(I* p) { return p->y; } // expected-error {{'y' is a private member of 'test1::E::I'}}
  };
}

// RUN: %clang_cc1 -fopenmp -fsyntax-only -verify %s

void f1() {
#pragma omp parallel
  class O {
    // expected-error@+1 {{templates cannot be declared inside of a local class}}
    template <class T> class I {
      void bar(bool b = true);
    };
    I<int> bar; // expected-error {{no template named 'I'}}
  };
}

void f2() {
#pragma omp parallel
  class O {
    class P {
      // expected-error@+1 {{templates cannot be declared inside of a local class}}
      template <class T> class I {
        void bar(bool b = true);
      };
      I<int> bar; // expected-error {{no template named 'I'}}
    };
  };
}

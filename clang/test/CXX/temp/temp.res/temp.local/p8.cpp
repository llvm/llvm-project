// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics

namespace N { 
  enum { C };
  template<class T> class B {
    void f(T);
  }; 
}

template<class C> void N::B<C>::f(C) {
  C b;
}

namespace N {
  enum { D };
  namespace M {
    enum { C , D };
    template<typename C> class X {
      template<typename U> void f(C, U);

      template<typename D> void g(C, D) {
        C c;
        D d;
      }
    };

    struct Y {
      template<typename U> void f(U);      
    };
  }

  struct Y {
    template<typename D> void f(D);
  };
}

template<typename C> 
template<typename D>
void N::M::X<C>::f(C, D) {
  C c;
  D d;
}

template<typename C>
void N::M::Y::f(C) {
  C c;
}

template<typename D> 
void N::Y::f(D) {
  D d;
}


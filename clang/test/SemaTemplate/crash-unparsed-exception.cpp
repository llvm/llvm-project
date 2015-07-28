// RUN: %clang_cc1 -fsyntax-only -std=c++11 -verify -fcxx-exceptions -fexceptions %s

struct A {
  virtual ~A();
};
template <class>
struct B {};
struct C {
  template <typename>
  struct D {
    ~D() throw();
  };
  struct E : A {
    D<int> d; //expected-error{{exception specification is not available until end of class definition}}
  };
  B<int> b; //expected-note{{in instantiation of template class 'B<int>' requested here}}
};

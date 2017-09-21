// RUN: %clang_cc1 -fsyntax-only -std=c++11 -verify -fcxx-exceptions -fexceptions %s
// expected-no-diagnostics

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
    D<int> d;
  };
  B<int> b;
};

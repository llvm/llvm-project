// RUN: %clang_cc1 -fsyntax-only -Wall -verify %s
// expected-no-diagnostics
template<typename a> struct A {
  template <typename b> struct B {
    template <typename c> struct C {
      template <typename d> struct D {
        template <typename e> struct E {
          e field;
          E() : field(0) {
            d v1 = 4;
            c v2 = v1 * v1;
            b v3 = 8;
            a v4 = v3 * v3;
            field += v2 + v4;
          }
        };
      };
    };
  };
};

A<int>::B<int>::C<int>::D<int>::E<int> global;

// PR5352
template <typename T>
class Foo {
public:
  Foo() {}
  
  struct Bar {
    T value;
  };
  
  Bar u;
};

template class Foo<int>;

// RUN: %clang_cc1 -std=c++23 -fsyntax-only -verify %s

struct A {
  union {
    int n = 0;
    int m;
  };
};
const A a;

struct B {
  union {
    struct {
      int n = 5;
      int m;
    };
  };
};
const B b; // expected-error {{default initialization of an object of const type 'const B' without a user-provided default constructor}}

struct S {
  int i;
  int j;
};

struct T {
  T() = default;
};

struct C {
  union {
    S s;
  };
};

struct D {
  union {
    T s;
  };
};

const C c; // expected-error {{default initialization of an object of const type 'const C' without a user-provided default constructor}}
const D d; // expected-error {{default initialization of an object of const type 'const D' without a user-provided default constructor}}

struct E {
  union {
    int n;
    int m=0;
  };
};
const E e;

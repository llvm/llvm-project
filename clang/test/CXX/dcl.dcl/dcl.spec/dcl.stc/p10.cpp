// RUN: %clang_cc1 -fsyntax-only -verify %s

typedef const int T0;
typedef int& T1;

struct s0 {
  mutable const int f0; // expected-error{{'mutable' and 'const' cannot be mixed}}
  mutable T0 f1; // expected-error{{'mutable' and 'const' cannot be mixed}}
  mutable int &f2; // expected-error{{'mutable' cannot be applied to references}}
  mutable T1 f3; // expected-error{{'mutable' cannot be applied to references}}
  mutable struct s1 {}; // expected-error{{'mutable' can only be applied to member variables}}
  mutable void im0(); // expected-error{{'mutable' cannot be applied to functions}}
};

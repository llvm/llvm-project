// RUN: %clang_cc1 -verify -fsyntax-only -fcomplete-member-pointers %s

struct S; // expected-note {{forward declaration of 'S'}}
typedef int S::*t;
t foo; // expected-error {{member pointer has incomplete base type 'S'}}

struct S2 {
  int S2::*foo;
};
int S2::*bar;

template <typename T>
struct S3 {
  int T::*foo;
};

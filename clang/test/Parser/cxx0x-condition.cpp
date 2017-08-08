// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s

struct S { S(int); operator bool(); };

void f() {
  int a;
  typedef int n;

  while (a) ;
  while (int x) ; // expected-error {{variable declaration in condition must have an initializer}}
  while (float x = 0) ;
  if (const int x = a) ; // expected-warning{{empty body}} expected-note{{put the semicolon on a separate line to silence this warning}}
  switch (int x = a+10) {}
  for (; int x = ++a; ) ;

  if (S(a)) {} // ok
  if (S(a) = 0) {} // ok
  if (S(a) == 0) {} // ok

  if (S(n)) {} // expected-error {{unexpected type name 'n': expected expression}}
  if (S(n) = 0) {} // ok
  if (S(n) == 0) {} // expected-error {{unexpected type name 'n': expected expression}}

  if (S b(a)) {} // expected-error {{variable declaration in condition cannot have a parenthesized initializer}}

  if (S b(n)) {} // expected-error {{a function type is not allowed here}}
  if (S b(n) = 0) {} // expected-error {{a function type is not allowed here}}
  if (S b(n) == 0) {} // expected-error {{a function type is not allowed here}}

  S s(a);
  if (S{s}) {} // ok
  if (S a{s}) {} // ok
  if (S a = {s}) {} // ok
  if (S a == {s}) {} // expected-error {{did you mean '='?}}

  if (S(b){a}) {} // ok
  if (S(b) = {a}) {} // ok
}

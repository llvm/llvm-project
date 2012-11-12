// RUN: %clang_cc1 %s -fsyntax-only -verify
// RUN: cp %s %t
// RUN: not %clang_cc1 -x c++ -fixit %t
// RUN: %clang_cc1 -x c++ %t

struct S { static int a,b,c;};
int S::(a);  // expected-error{{unexpected parenthesis after '::'}}
int S::(b;  // expected-error{{unexpected parenthesis after '::'}}
int S::c;
int S::(*d);  // expected-error{{unexpected parenthesis after '::'}}
int S::(*e;  // expected-error{{unexpected parenthesis after '::'}}
int S::*f;
int g = S::(a);  // expected-error{{unexpected parenthesis after '::'}}
int h = S::(b;  // expected-error{{unexpected parenthesis after '::'}}
int i = S::c;

void foo() {
  int a;
  a = ::(g);  // expected-error{{unexpected parenthesis after '::'}}
  a = ::(h;  // expected-error{{unexpected parenthesis after '::'}}
  a = ::i;
}

// RUN: %clang_cc1 %s -fsyntax-only -verify -DPR21815
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

#ifdef PR21815
// expected-error@+2{{C++ requires a type specifier for all declarations}}
// expected-error@+1{{expected unqualified-id}}
a (::( ));

::((c )); // expected-error{{expected unqualified-id}}
#endif

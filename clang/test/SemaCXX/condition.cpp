// RUN: %clang_cc1 -fsyntax-only -verify %s 

void test() {
  int x;
  if (x) ++x;
  if (int x=0) ++x;

  typedef int arr[10];
  while (arr x=0) ; // expected-error {{an array type is not allowed here}} expected-error {{array initializer must be an initializer list}}
  while (int f()=0) ; // expected-error {{a function type is not allowed here}}

  struct S {} s;
  if (s) ++x; // expected-error {{value of type 'struct S' is not contextually convertible to 'bool'}}
  while (struct S x=s) ; // expected-error {{value of type 'struct S' is not contextually convertible to 'bool'}}
  do ; while (s); // expected-error {{value of type 'struct S' is not contextually convertible to 'bool'}}
  for (;s;) ; // expected-error {{value of type 'struct S' is not contextually convertible to 'bool'}}
  switch (s) {} // expected-error {{statement requires expression of integer type ('struct S' invalid)}}

  while (struct NewS *x=0) ;
  while (struct S {} *x=0) ; // expected-error {{types may not be defined in conditions}}
  while (struct {} *x=0) ; // expected-error {{types may not be defined in conditions}}
  switch (enum {E} x=0) ; // expected-error {{types may not be defined in conditions}} expected-error {{cannot initialize}} \
  // expected-warning{{enumeration value 'E' not handled in switch}} expected-warning {{switch statement has empty body}} \
  // expected-note{{put the semicolon on a separate line}}

  if (int x=0) { // expected-note 2 {{previous definition is here}}
    int x;  // expected-error {{redefinition of 'x'}}
  }
  else
    int x;  // expected-error {{redefinition of 'x'}}
  while (int x=0) int x; // expected-error {{redefinition of 'x'}} expected-note {{previous definition is here}}
  while (int x=0) { int x; } // expected-error {{redefinition of 'x'}} expected-note {{previous definition is here}}
  for (int x; int x=0; ) ; // expected-error {{redefinition of 'x'}} expected-note {{previous definition is here}}
  for (int x; ; ) int x; // expected-error {{redefinition of 'x'}} expected-note {{previous definition is here}}
  for (; int x=0; ) int x; // expected-error {{redefinition of 'x'}} expected-note {{previous definition is here}}
  for (; int x=0; ) { int x; } // expected-error {{redefinition of 'x'}} expected-note {{previous definition is here}}
  switch (int x=0) { default: int x; } // expected-error {{redefinition of 'x'}} expected-note {{previous definition is here}}
}

int* get_int_ptr();

void test2() {
  float *ip;
  if (int *ip = ip) {
  }
}

// Make sure we do function/array decay.
void test3() {
  if ("help")
    (void) 0;

  if (test3) // expected-warning {{address of function 'test3' will always evaluate to 'true'}} \
                expected-note {{prefix with the address-of operator to silence this warning}}
    (void) 0;
}

void test4(bool (&x)(void)) {
  while (x);
}

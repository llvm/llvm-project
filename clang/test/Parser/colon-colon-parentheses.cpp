// RUN: %clang_cc1 %s -verify -fno-spell-checking

struct S { static int a,b,c;};
int S::(a);  // expected-error{{expected unqualified-id}}
int S::(b;  // expected-error{{expected unqualified-id}}
        );
int S::c;
int S::(*d);  // expected-error{{expected unqualified-id}}
int S::(*e;  // expected-error{{expected unqualified-id}}
        );
int S::*f;
int g = S::(a);  // expected-error {{expected unqualified-id}} expected-error {{use of undeclared identifier 'a'}}
int h = S::(b;  // expected-error {{expected unqualified-id}} expected-error {{use of undeclared identifier 'b'}}
            );
int i = S::c;

void foo() {
  int a;
  a = ::(g);  // expected-error{{expected unqualified-id}}
  a = ::(h;  // expected-error{{expected unqualified-id}}
  a = ::i;
}

// The following tests used to be crash bugs.

// PR21815
// expected-error@+2{{C++ requires a type specifier for all declarations}}
// expected-error@+1{{expected unqualified-id}}
a (::( ));

::((c )); // expected-error{{expected unqualified-id}}

// PR26623
int f1(::(B) p); // expected-error {{expected unqualified-id}} expected-error {{use of undeclared identifier 'B'}}

int f2(::S::(C) p); // expected-error {{expected unqualified-id}} expected-error {{use of undeclared identifier 'C'}}

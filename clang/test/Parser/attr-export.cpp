// RUN: %clang_cc1 -triple s390x-ibm-zos -fsyntax-only -verify %s

int _Export *p; // expected-error {{expected unqualified-id}}
int * _Export *p2; // expected-error {{expected unqualified-id}}
int *pr3 _Export; // expected-error {{expected ';' after top level declarator}}

int f _Export (); // expected-error {{expected ';' after top level declarator}}
int f2() _Export; // expected-error {{expected function body after function declarator}}


int * _Export pg;

int (*_Export fp)();

int _Export fg();

struct S {};

int S::* _Export mp1;
int _Export S::* mp2; // expected-error {{expected unqualified-id}}

struct _Export S1 {};
struct S2 _Export {}; // expected-error {{expected unqualified-id}}
_Export struct S3 {}; // expected-error {{expected unqualified-id}}


// RUN: %clang_cc1 -fsyntax-only -Wno-strict-prototypes -fsuppress-conflicting-types -verify %s

void blapp(int);
void blapp() {}

void foo(int);
void foo();
void foo() {}

// Maintain preexisting released clang behavior that catches conflicting type errors.
void yarp(int, ...); // expected-note {{previous}}
void yarp();         // expected-error {{conflicting types for 'yarp'}}

void blarg(int, ...); // expected-note {{previous}}
void blarg() {}       // expected-error {{conflicting types for 'blarg'}}

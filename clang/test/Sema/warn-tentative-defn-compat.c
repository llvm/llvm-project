// RUN: %clang_cc1 -fsyntax-only -verify -Wtentative-definition-compat %s
// RUN: %clang_cc1 -fsyntax-only -verify -Wc++-compat %s
// RUN: %clang_cc1 -fsyntax-only -verify=good %s
// RUN: %clang_cc1 -fsyntax-only -verify=cxx -x c++ %s
// good-no-diagnostics

int i;      // expected-note {{previous declaration is here}} \
               cxx-note {{previous definition is here}}
int i;      // expected-warning {{duplicate declaration of 'i' is invalid in C++}} \
               cxx-error {{redefinition of 'i'}}

int j = 12; // expected-note {{previous declaration is here}} \
               cxx-note {{previous definition is here}}
int j;      // expected-warning {{duplicate declaration of 'j' is invalid in C++}} \
               cxx-error {{redefinition of 'j'}}

int k;      // expected-note {{previous declaration is here}} \
               cxx-note {{previous definition is here}}
int k = 12; // expected-warning {{duplicate declaration of 'k' is invalid in C++}} \
               cxx-error {{redefinition of 'k'}}

// Cannot have two declarations with initializers, that is a redefinition in
// both C and C++. However, C++ does have a different definition of what makes
// a declaration a definition.
extern const int a;
const int a = 12; // Okay in C and C++

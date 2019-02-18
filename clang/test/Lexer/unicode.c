// RUN: %clang_cc1 -fsyntax-only -verify -x c -std=c11 %s
// RUN: %clang_cc1 -fsyntax-only -verify -x c++ -std=c++11 %s
// RUN: %clang_cc1 -E -DPP_ONLY=1 %s -o %t
// RUN: FileCheck --strict-whitespace --input-file=%t %s

// This file contains Unicode characters; please do not "fix" them!

extern int x; // expected-warning {{treating Unicode character as whitespace}}
extern int　x; // expected-warning {{treating Unicode character as whitespace}}

// CHECK: extern int {{x}}
// CHECK: extern int　{{x}}

#pragma mark ¡Unicode!

#define COPYRIGHT Copyright © 2012
#define XSTR(X) #X
#define STR(X) XSTR(X)

static const char *copyright = STR(COPYRIGHT); // no-warning
// CHECK: static const char *copyright = "Copyright © {{2012}}";

#if PP_ONLY
COPYRIGHT
// CHECK: Copyright © {{2012}}
CHECK: The preprocessor should not complain about Unicode characters like ©.
#endif

// A 🌹 by any other name....
extern int 🌹;
int 🌵(int 🌻) { return 🌻+ 1; }
int main () {
  int 🌷 = 🌵(🌹);
  return 🌷;
}

int n; = 3; // expected-warning {{treating Unicode character <U+037E> as identifier character rather than as ';' symbol}}
int *n꞉꞉v = &n;; // expected-warning 2{{treating Unicode character <U+A789> as identifier character rather than as ':' symbol}}
                 // expected-warning@-1 {{treating Unicode character <U+037E> as identifier character rather than as ';' symbol}}
int v＝［＝］（auto）｛return～x；｝（）; // expected-warning 12{{treating Unicode character}}

int ⁠x﻿x‍;
// expected-warning@-1 {{identifier contains Unicode character <U+2060> that is invisible in some environments}}
// expected-warning@-2 {{identifier contains Unicode character <U+FEFF> that is invisible in some environments}}
// expected-warning@-3 {{identifier contains Unicode character <U+200D> that is invisible in some environments}}
int foo​bar = 0; // expected-warning {{identifier contains Unicode character <U+200B> that is invisible in some environments}}
int x = foobar; // expected-error {{undeclared identifier}}

int ∣foo; // expected-error {{non-ASCII character}}
#ifndef PP_ONLY
#define ∶ x // expected-error {{macro name must be an identifier}}
#endif

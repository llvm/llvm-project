// RUN: %clang_cc1 -fsyntax-only -verify -x c -std=c11 %s
// RUN: %clang_cc1 -fsyntax-only -verify -x c++ -std=c++11 %s
// RUN: %clang_cc1 -std=c99 -E -DPP_ONLY=1 %s | FileCheck %s --strict-whitespace
// RUN: %clang_cc1 -E -DPP_ONLY=1 %s | FileCheck %s --strict-whitespace

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
CHECK : The preprocessor should not complain about Unicode characters like ©.
#endif

        int _;

#ifdef __cplusplus

extern int ༀ;
extern int 𑩐;
extern int 𐠈;
extern int ꙮ;
extern int  \u1B4C;     // BALINESE LETTER ARCHAIC JNYA - Added in Unicode 14
extern int  \U00016AA2; // TANGSA LETTER GA - Added in Unicode 14
// This character doesn't have the XID_Start property
extern int  \U00016AC0; // TANGSA DIGIT ZERO  // expected-error {{expected unqualified-id}}
extern int _\U00016AC0; // TANGSA DIGIT ZERO

extern int 🌹; // expected-error {{unexpected character <U+1F339>}} \
                  expected-warning {{declaration does not declare anything}}

extern int 👷; // expected-error {{unexpected character <U+1F477>}} \
                  expected-warning {{declaration does not declare anything}}

extern int 👷‍♀; // expected-warning {{declaration does not declare anything}} \
                  expected-error {{unexpected character <U+1F477>}} \
                  expected-error {{unexpected character <U+200D>}} \
                  expected-error {{unexpected character <U+2640>}}
#else

// A 🌹 by any other name....
extern int 🌹;
int 🌵(int 🌻) { return 🌻+ 1; }
int main (void) {
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

int ∣foo; // expected-error {{unexpected character <U+2223>}}
#ifndef PP_ONLY
#define ∶ x // expected-error {{macro name must be an identifier}}
#endif

#endif

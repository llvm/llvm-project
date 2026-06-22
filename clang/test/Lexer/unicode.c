// RUN: %clang_cc1 -fsyntax-only -verify -x c -std=c11 %s
// RUN: %clang_cc1 -fsyntax-only -verify=expected,c2x -x c -std=c2x %s
// RUN: %clang_cc1 -fsyntax-only -verify=expected,cxx -x c++ -std=c++11 %s
// RUN: %clang_cc1 -std=c99 -E -DPP_ONLY=1 %s | FileCheck %s --strict-whitespace
// RUN: %clang_cc1 -E -DPP_ONLY=1 %s | FileCheck %s --strict-whitespace
// UNSUPPORTED: system-zos

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

int a;

extern int X\UAAAAAAAA; // expected-error {{not allowed in an identifier}}
int Y = '\UAAAAAAAA'; // expected-error {{invalid universal character}}

#if defined(__cplusplus) || (defined(__STDC_VERSION__) && __STDC_VERSION__ >= 202311L)

extern int ༀ;
extern int 𑩐;
extern int 𐠈;
extern int ꙮ;
extern int \u1B4C;     // BALINESE LETTER ARCHAIC JNYA - Added in Unicode 14
extern int \U00016AA2; // TANGSA LETTER GA - Added in Unicode 14
extern int \U0001E4D0; // 𞓐 NAG MUNDARI LETTER O - Added in Unicode 15
extern int \u{2EBF0};  // CJK UNIFIED IDEOGRAPH-2EBF0 - Added in Unicode 15.1
extern int \u{10D5A};   // GARAY CAPITAL LETTER DA - Added in Unicode 16.0
extern int \u{16EBE};   // BERIA ERFE SMALL LETTER EH - Added in Unicode 17.0
extern int \u{16D80};   // CHISOI LETTER A - Added in Unicode 18.0
extern int a\N{TANGSA LETTER GA};
extern int a\N{TANGSALETTERGA}; // expected-error {{'TANGSALETTERGA' is not a valid Unicode character name}} \
                                // expected-error {{expected ';' after top level declarator}} \
                                // expected-note {{characters names in Unicode escape sequences are sensitive to case and whitespace}}

extern int 𝛛; // expected-warning {{mathematical notation character <U+1D6DB> in an identifier is a Clang extension}}
extern int ₉; // expected-error {{character <U+2089> not allowed at the start of an identifier}} \\
                 expected-warning {{declaration does not declare anything}}

extern int a\N{PICKLE}; // expected-error {{character <U+1FADD> not allowed in an identifier}}

int a¹b₍₄₂₎∇; // expected-warning 6{{mathematical notation character}}

int \u{221E} = 1; // expected-warning {{mathematical notation character}}
int \N{MATHEMATICAL SANS-SERIF BOLD ITALIC PARTIAL DIFFERENTIAL} = 1;
                 // expected-warning@-1 {{mathematical notation character}}

int a\N{SUBSCRIPT EQUALS SIGN} = 1; // expected-warning {{mathematical notation character}}

// This character doesn't have the XID_Start property
extern int  \U00016AC0; // TANGSA DIGIT ZERO  // cxx-error {{expected unqualified-id}} \
                                              // c2x-error {{expected identifier or '('}}

extern int 🌹; // expected-error {{unexpected character <U+1F339>}} \
                  expected-warning {{declaration does not declare anything}}

extern int 🫎;   // MOOSE (Unicode 15) \
                // expected-error {{unexpected character <U+1FACE>}} \
                   expected-warning {{declaration does not declare anything}}

extern int 👷; // expected-error {{unexpected character <U+1F477>}} \
                  expected-warning {{declaration does not declare anything}}

extern int 👷‍♀; // expected-warning {{declaration does not declare anything}} \
                  expected-error {{unexpected character <U+1F477>}} \
                  expected-error {{character <U+200D> not allowed at the start of an identifier}} \
                  expected-error {{unexpected character <U+2640>}}
#else

// A 🌹 by any other name....
extern int 🌹;
int 🌵(int 🌻) { return 🌻+ 1; }
int main (void) {
  int 🌷 = 🌵(🌹);
  return 🌷;
}

int n; = 3; // expected-warning {{treating Unicode character <U+037E> as an identifier character rather than as ';' symbol}}
int *n꞉꞉v = &n;; // expected-warning 2{{treating Unicode character <U+A789> as an identifier character rather than as ':' symbol}}
                 // expected-warning@-1 {{treating Unicode character <U+037E> as an identifier character rather than as ';' symbol}}
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

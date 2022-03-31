// RUN: %clang_cc1 -fsyntax-only -verify -fdouble-square-bracket-attributes %s

int var __attribute__((internal_linkage));
int var2 __attribute__((internal_linkage,common)); // expected-error{{'common' and 'internal_linkage' attributes are not compatible}} \
                                                   // expected-note{{conflicting attribute is here}}
int var3 __attribute__((common,internal_linkage)); // expected-error{{'internal_linkage' and 'common' attributes are not compatible}} \
                                                   // expected-note{{conflicting attribute is here}}

int var4 __attribute__((common)); // expected-note{{previous declaration is here}} expected-note{{conflicting attribute is here}}
int var4 __attribute__((internal_linkage)); // expected-error{{'internal_linkage' and 'common' attributes are not compatible}} \
                                            // expected-error{{'internal_linkage' attribute does not appear on the first declaration}}

int var5 __attribute__((internal_linkage)); // expected-note{{conflicting attribute is here}}
int var5 __attribute__((common)); // expected-error{{'common' and 'internal_linkage' attributes are not compatible}}

__attribute__((internal_linkage)) int f(void) {}
struct __attribute__((internal_linkage)) S { // expected-warning{{'internal_linkage' attribute only applies to variables, functions, and classes}}
};

__attribute__((internal_linkage("foo"))) int g(void) {} // expected-error{{'internal_linkage' attribute takes no arguments}}

int var6 [[clang::internal_linkage]];
int var7 [[clang::internal_linkage]] __attribute__((common)); // expected-error{{'internal_linkage' and 'common' attributes are not compatible}} \
                                                   // expected-note{{conflicting attribute is here}}
__attribute__((common)) int var8 [[clang::internal_linkage]]; // expected-error{{'internal_linkage' and 'common' attributes are not compatible}} \
                                                   // expected-note{{conflicting attribute is here}}

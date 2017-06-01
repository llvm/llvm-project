// RUN: %clang_cc1 -fsyntax-only -verify %s

int var __attribute__((internal_linkage));
int var2 __attribute__((internal_linkage,common)); // expected-error{{'internal_linkage' and 'common' attributes are not compatible}} \
                                                   // expected-note{{conflicting attribute is here}}
int var3 __attribute__((common,internal_linkage)); // expected-error{{'common' and 'internal_linkage' attributes are not compatible}} \
                                                   // expected-note{{conflicting attribute is here}}

int var4 __attribute__((common)); // expected-error{{'common' and 'internal_linkage' attributes are not compatible}} \
// expected-note{{previous definition is here}}
int var4 __attribute__((internal_linkage)); // expected-note{{conflicting attribute is here}} \
// expected-error{{'internal_linkage' attribute does not appear on the first declaration of 'var4'}}

int var5 __attribute__((internal_linkage)); // expected-error{{'internal_linkage' and 'common' attributes are not compatible}}
int var5 __attribute__((common)); // expected-note{{conflicting attribute is here}}

__attribute__((internal_linkage)) int f() {}
struct __attribute__((internal_linkage)) S { // expected-warning{{'internal_linkage' attribute only applies to variables and functions}}
};

__attribute__((internal_linkage("foo"))) int g() {} // expected-error{{'internal_linkage' attribute takes no arguments}}

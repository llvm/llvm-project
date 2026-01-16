// RUN: %clang_cc1 -fsyntax-only -Wattribute-preprocessor-tokens -verify %s
// RUN: %clang_cc1 -Wattribute-preprocessor-tokens -E %s | FileCheck %s
// RUN: %clang_cc1 -x c -fsyntax-only -verify=c %s
// RUN: %clang_cc1 -x c -E %s | FileCheck %s

#define ATTR_STR(X) [[clang::annotate(#X)]]
#define ATTR_PASTE(X, Y) [[clang::annotate("test", X ## Y)]]

[[clang::assume(#)]] void f1();           // c-error {{expected expression}} \
                                          // expected-warning {{'#' is not allowed in an attribute argument list}}

[[clang::assume(##)]] void f2();          // c-error {{expected expression}} \
                                          // expected-warning {{'##' is not allowed in an attribute argument list}}

[[clang::assume(1#2#3)]] void f3();       // c-error {{use of this expression in an 'assume' attribute requires parentheses}} \
                                          // c-error {{expected ')'}} \
                                          // c-note {{to match this '('}} \
                                          // expected-warning {{'#' is not allowed in an attribute argument list}} \
                                          // expected-warning {{'#' is not allowed in an attribute argument list}}

[[unknown::unknown(#)]] void f4();        // c-warning {{unknown attribute 'unknown::unknown' ignored}} \
                                          // expected-warning {{'#' is not allowed in an attribute argument list}}

[[unknown::unknown(##)]] void f5();       // c-warning {{unknown attribute 'unknown::unknown' ignored}} \
                                          // expected-warning {{'##' is not allowed in an attribute argument list}}

[[unknown::unknown(1#2#3)]] void f6();    // c-warning {{unknown attribute 'unknown::unknown' ignored}} \
                                          // expected-warning {{'#' is not allowed in an attribute argument list}} \
                                          // expected-warning {{'#' is not allowed in an attribute argument list}}

[[clang::assume(%:)]] void f7();          // c-error {{expected expression}} \
                                          // expected-warning {{'%:' is not allowed in an attribute argument list}}


[[clang::assume(%:%:)]] void f8();        // c-error {{expected expression}} \
                                          // expected-warning {{'%:%:' is not allowed in an attribute argument list}}

[[clang::assume(1%:2%:3)]] void f9();     // c-error {{use of this expression in an 'assume' attribute requires parentheses}} \
                                          // c-error {{expected ')'}} \
                                          // c-note {{to match this '('}} \
                                          // expected-warning {{'%:' is not allowed in an attribute argument list}} \
                                          // expected-warning {{'%:' is not allowed in an attribute argument list}}

[[unknown::unknown(%:)]] void f10();      // c-warning {{unknown attribute 'unknown::unknown' ignored}} \
                                          // expected-warning {{'%:' is not allowed in an attribute argument list}}

[[unknown::unknown(%:%:)]] void f11();    // c-warning {{unknown attribute 'unknown::unknown' ignored}} \
                                          // expected-warning {{'%:%:' is not allowed in an attribute argument list}}

[[unknown::unknown(1%:2%:3)]] void f12(); // c-warning {{unknown attribute 'unknown::unknown' ignored}} \
                                          // expected-warning {{'%:' is not allowed in an attribute argument list}} \
                                          // expected-warning {{'%:' is not allowed in an attribute argument list}}

ATTR_STR(stringify) void f13();
// CHECK: {{\[\[}}clang{{::}}annotate("stringify"){{\]\]}} void f13();

ATTR_PASTE(1, 2) void f14();
// CHECK: {{\[\[}}clang{{::}}annotate("test", 12){{\]\]}} void f14();

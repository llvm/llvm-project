// RUN: %clang_cc1 -fsyntax-only -verify %s

#pragma clang attribute pop // expected-error {{'#pragma clang attribute pop' with no matching '#pragma clang attribute push'}}

// Don't verify unused attributes.
#pragma clang attribute push (__attribute__((annotate)), apply_to = function) // expected-warning {{unused attribute 'annotate' in '#pragma clang attribute push' region}}
#pragma clang attribute pop // expected-note {{'#pragma clang attribute push' regions ends here}}

// Ensure we only report any errors once.
#pragma clang attribute push (__attribute__((annotate)), apply_to = function) // expected-error 4 {{'annotate' attribute takes one argument}}

void test5_begin(); // expected-note {{when applied to this declaration}}
void test5_1(); // expected-note {{when applied to this declaration}}

#pragma clang attribute push (__attribute__((annotate())), apply_to = function) // expected-error 2 {{'annotate' attribute takes one argument}}

void test5_2(); // expected-note 2 {{when applied to this declaration}}

#pragma clang attribute push (__attribute__((annotate("hello", "world"))), apply_to = function) // expected-error {{'annotate' attribute takes one argument}}

void test5_3(); // expected-note 3 {{when applied to this declaration}}

#pragma clang attribute pop
#pragma clang attribute pop
#pragma clang attribute pop

// Verify that the warnings are reported for each receiver declaration

#pragma clang attribute push (__attribute__((optnone)), apply_to = function) // expected-note 2 {{conflicting attribute is here}}

__attribute__((always_inline)) void optnone1() { } // expected-warning {{'always_inline' attribute ignored}}
// expected-note@-1 {{when applied to this declaration}}

void optnone2() { }

__attribute__((always_inline)) void optnone3() { } // expected-warning {{'always_inline' attribute ignored}}
// expected-note@-1 {{when applied to this declaration}}

#pragma clang attribute pop

#pragma clang attribute push (__attribute__((annotate())), apply_to = function) // expected-error{{'annotate' attribute takes one argument}}
#pragma clang attribute (__attribute__((annotate())), apply_to = function) // expected-error{{'annotate' attribute takes one argument}}

void fun(); // expected-note 2 {{when applied to this declaration}}

#pragma clang attribute pop
#pragma clang attribute pop // expected-error{{'#pragma clang attribute pop' with no matching '#pragma clang attribute push'}}


#pragma clang attribute push
#pragma clang attribute (__attribute__((annotate())), apply_to = function) // expected-error 2 {{'annotate' attribute takes one argument}}

void fun2(); // expected-note {{when applied to this declaration}}

#pragma clang attribute push (__attribute__((annotate())), apply_to = function) // expected-error{{'annotate' attribute takes one argument}}
void fun3(); // expected-note 2 {{when applied to this declaration}}
#pragma clang attribute pop

#pragma clang attribute pop
#pragma clang attribute pop // expected-error{{'#pragma clang attribute pop' with no matching '#pragma clang attribute push'}}

#pragma clang attribute (__attribute__((annotate)), apply_to = function) // expected-error{{'#pragma clang attribute' attribute with no matching '#pragma clang attribute push}}

#pragma clang attribute push ([[]], apply_to = function) // A noop

#pragma clang attribute pop // expected-error {{'#pragma clang attribute pop' with no matching '#pragma clang attribute push'}}

#pragma clang attribute push (__attribute__((annotate("func"))), apply_to = function) // expected-error {{unterminated '#pragma clang attribute push' at end of file}}

void function();

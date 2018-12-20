// RUN: %clang_cc1 -fsyntax-only -verify %s

#pragma clang attribute MyNamespace.push (__attribute__((annotate)), apply_to=function) // expected-error 2 {{'annotate' attribute}}

int some_func(); // expected-note{{when applied to this declaration}}

#pragma clang attribute pop // expected-error{{'#pragma clang attribute pop' with no matching '#pragma clang attribute push'}}
#pragma clang attribute NotMyNamespace.pop // expected-error{{'#pragma clang attribute NotMyNamespace.pop' with no matching '#pragma clang attribute NotMyNamespace.push'}}

#pragma clang attribute MyOtherNamespace.push (__attribute__((annotate)), apply_to=function) // expected-error 2 {{'annotate' attribute}}

int some_other_func(); // expected-note 2 {{when applied to this declaration}}

// Out of order!
#pragma clang attribute MyNamespace.pop

int some_other_other_func(); // expected-note 1 {{when applied to this declaration}}

#pragma clang attribute MyOtherNamespace.pop

#pragma clang attribute Misc. () // expected-error{{namespace can only apply to 'push' or 'pop' directives}} expected-note {{omit the namespace to add attributes to the most-recently pushed attribute group}}

#pragma clang attribute Misc push // expected-error{{expected '.' after pragma attribute namespace 'Misc'}}

// Test how pushes with namespaces interact with pushes without namespaces.

#pragma clang attribute Merp.push (__attribute__((annotate)), apply_to=function) // expected-error{{'annotate' attribute}}
#pragma clang attribute push (__attribute__((annotate)), apply_to=function) // expected-warning {{unused attribute}}
#pragma clang attribute pop // expected-note{{ends here}}
int test(); // expected-note{{when applied to this declaration}}
#pragma clang attribute Merp.pop

#pragma clang attribute push (__attribute__((annotate)), apply_to=function) // expected-warning {{unused attribute}}
#pragma clang attribute Merp.push (__attribute__((annotate)), apply_to=function) // expected-error{{'annotate' attribute}}
#pragma clang attribute pop // expected-note{{ends here}}
int test2(); // expected-note{{when applied to this declaration}}
#pragma clang attribute Merp.pop

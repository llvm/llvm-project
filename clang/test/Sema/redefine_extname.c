// RUN: %clang_cc1 -triple=x86_64-unknown-linux -Wpragmas -verify %s

// Check that pragma redefine_extname applies to external declarations only.
#pragma redefine_extname foo_static bar_static
static int foo_static() { return 1; } // expected-warning {{#pragma redefine_extname is applicable to external C declarations only; not applied to function 'foo_static'}}


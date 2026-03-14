// RUN: %clang_cc1 -triple=x86_64-unknown-linux -Wpragmas -verify %s

// Check that pragma redefine_extname applies to C code only, and shouldn't be
// applied to C++.
#pragma redefine_extname foo_cpp bar_cpp
extern int foo_cpp() { return 1; } // expected-warning {{#pragma redefine_extname is applicable to external C declarations only; not applied to function 'foo_cpp'}}

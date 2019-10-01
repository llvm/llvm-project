// RUN: %clang_cc1 -verify -print-dependency-directives-minimized-source %s 2>&1

#define 0 0 // expected-error {{macro name must be an identifier}}

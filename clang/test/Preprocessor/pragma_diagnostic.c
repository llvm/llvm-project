// RUN: %clang_cc1 -fsyntax-only -verify -Wno-undef %s
// rdar://2362963

#if FOO    // ok.
#endif

#pragma GCC diagnostic warning "-Wundef"

#if FOO    // expected-warning {{'FOO' is not defined}}
#endif

#pragma GCC diagnostic ignored "-Wun" "def"

#if FOO    // ok.
#endif

#pragma GCC diagnostic error "-Wundef"

#if FOO    // expected-error {{'FOO' is not defined}}
#endif


#define foo error
#pragma GCC diagnostic foo "-Wundef"  // expected-warning {{pragma diagnostic expected 'error', 'warning', 'ignored', 'fatal', 'push', or 'pop'}}

#pragma GCC diagnostic error 42  // expected-warning {{unexpected token in pragma diagnostic}}

#pragma GCC diagnostic error "-Wundef" 42  // expected-warning {{unexpected token in pragma diagnostic}}
#pragma GCC diagnostic error "invalid-name"  // expected-warning {{pragma diagnostic expected option name (e.g. "-Wundef")}}

#pragma GCC diagnostic error "-Winvalid-name"  // expected-warning {{unknown warning group '-Winvalid-name', ignored}}


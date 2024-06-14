// RUN: %clang_cc1 -fsyntax-only -verify -Wno-undef %s
// RUN: %clang_cc1 -fsyntax-only -verify -Wno-undef -Wno-unknown-warning-option -DAVOID_UNKNOWN_WARNING %s
// RUN: %clang_cc1 -fsyntax-only -verify -Werror=undef -DINITIAL_UNDEF %s

#ifdef INITIAL_UNDEF
#if FOO    // expected-error {{'FOO' is not defined}}
#endif
#else
#if FOO    // ok.
#endif
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

#pragma GCC diagnostic error 42  // expected-error {{expected string literal in pragma diagnostic}}

#pragma GCC diagnostic error "-Wundef" 42  // expected-warning {{unexpected token in pragma diagnostic}}
#pragma GCC diagnostic error "invalid-name"  // expected-warning {{pragma diagnostic expected option name (e.g. "-Wundef")}}

#pragma GCC diagnostic error "-Winvalid-name"
#ifndef AVOID_UNKNOWN_WARNING
// expected-warning@-2 {{unknown warning group '-Winvalid-name', ignored}}
#endif

// From GH13920
#pragma clang diagnostic push ignored "-Wdeprecated-declarations" // expected-warning {{unexpected token in pragma diagnostic}}
#pragma clang diagnostic pop ignored "-Wdeprecated-declarations"  // expected-warning {{unexpected token in pragma diagnostic}}


// Testing pragma clang diagnostic with -Weverything
void ppo(void){} // First test that we do not diagnose on this.

#pragma clang diagnostic warning "-Weverything"
void ppp(void){} // expected-warning {{no previous prototype for function 'ppp'}}
// expected-note@-1{{declare 'static' if the function is not intended to be used outside of this translation unit}}

#pragma clang diagnostic ignored "-Weverything" // Reset it.
void ppq(void){}

#pragma clang diagnostic error "-Weverything" // Now set to error
void ppr(void){} // expected-error {{no previous prototype for function 'ppr'}}
// expected-note@-1{{declare 'static' if the function is not intended to be used outside of this translation unit}}

#pragma clang diagnostic warning "-Weverything" // Set to warning
void pps(void){} // expected-warning {{no previous prototype for function 'pps'}}
// expected-note@-1{{declare 'static' if the function is not intended to be used outside of this translation unit}}

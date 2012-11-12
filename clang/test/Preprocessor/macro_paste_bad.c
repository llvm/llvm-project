// RUN: %clang_cc1 -Eonly -verify -pedantic %s
// pasting ""x"" and ""+"" does not give a valid preprocessing token
#define XYZ  x ## + 
XYZ   // expected-error {{pasting formed 'x+', an invalid preprocessing token}}
#define XXYZ  . ## test
XXYZ   // expected-error {{pasting formed '.test', an invalid preprocessing token}}

// GCC PR 20077

#define a   a ## ## // expected-error {{'##' cannot appear at end of macro expansion}}
#define b() b ## ## // expected-error {{'##' cannot appear at end of macro expansion}}
#define c   c ##    // expected-error {{'##' cannot appear at end of macro expansion}}
#define d() d ##    // expected-error {{'##' cannot appear at end of macro expansion}}


#define e   ## ## e // expected-error {{'##' cannot appear at start of macro expansion}}
#define f() ## ## f // expected-error {{'##' cannot appear at start of macro expansion}}
#define g   ## g    // expected-error {{'##' cannot appear at start of macro expansion}}
#define h() ## h    // expected-error {{'##' cannot appear at start of macro expansion}}
#define i   ##      // expected-error {{'##' cannot appear at start of macro expansion}}
#define j() ##      // expected-error {{'##' cannot appear at start of macro expansion}}

// Invalid token pasting.
// PR3918

// When pasting creates poisoned identifiers, we error.
#pragma GCC poison BLARG
BLARG  // expected-error {{attempt to use a poisoned identifier}}
#define XX BL ## ARG
XX     // expected-error {{attempt to use a poisoned identifier}}

#define VA __VA_ ## ARGS__
int VA;   // expected-warning {{__VA_ARGS__ can only appear in the expansion of a C99 variadic macro}}


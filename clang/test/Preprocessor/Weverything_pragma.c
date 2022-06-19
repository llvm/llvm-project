// RUN: %clang_cc1 -Weverything   -fsyntax-only -verify %s

// Test that the pragma overrides command line option -Weverythings,

// a diagnostic with DefaultIgnore. This is part of a group 'unused-macro'
// but -Weverything forces it
#define UNUSED_MACRO1 1 // expected-warning{{macro is not used}}

void foo(void) // expected-warning {{no previous prototype for function}}
// expected-note@-1{{declare 'static' if the function is not intended to be used outside of this translation unit}}
{
 // A diagnostic without DefaultIgnore, and not part of a group.
 (void) 'ab'; // expected-warning {{multi-character character constant}}

#pragma clang diagnostic warning "-Weverything" // Should not change anyhting.
#define UNUSED_MACRO2 1 // expected-warning{{macro is not used}}
 (void) 'cd'; // expected-warning {{multi-character character constant}}

#pragma clang diagnostic ignored "-Weverything" // Ignore warnings now.
#define UNUSED_MACRO2 1 // no warning
 (void) 'ef'; // no warning here

#pragma clang diagnostic warning "-Weverything" // Revert back to warnings.
#define UNUSED_MACRO3 1 // expected-warning{{macro is not used}}
 (void) 'gh'; // expected-warning {{multi-character character constant}}

#pragma clang diagnostic error "-Weverything"  // Give errors now.
#define UNUSED_MACRO4 1 // expected-error{{macro is not used}}
 (void) 'ij'; // expected-error {{multi-character character constant}}
}

// RUN: %clang_cc1 -verify -std=c99 %s
// RUN: %clang_cc1 -verify -std=c99 -fno-dollars-in-identifiers %s

/* WG14 N717: Clang 17
 * Extended identifiers
 */

// Used as a sink for UCNs.
#define M(arg)

// C99 6.4.3p1 specifies the grammar for UCNs. A \u must be followed by exactly
// four hex digits, and \U must be followed by exactly eight.
M(\u1)    // expected-warning {{incomplete universal character name; treating as '\' followed by identifier}}
M(\u12)   // expected-warning {{incomplete universal character name; treating as '\' followed by identifier}}
M(\u123)  // expected-warning {{incomplete universal character name; treating as '\' followed by identifier}}
M(\u1234) // Okay
M(\u12345)// Okay, two tokens (UCN followed by 5)

M(\U1)         // expected-warning {{incomplete universal character name; treating as '\' followed by identifier}}
M(\U12)        // expected-warning {{incomplete universal character name; treating as '\' followed by identifier}}
M(\U123)       // expected-warning {{incomplete universal character name; treating as '\' followed by identifier}}
M(\U1234)      // expected-warning {{incomplete universal character name; treating as '\' followed by identifier}} \
                  expected-note {{did you mean to use '\u'?}}
M(\U12345)     // expected-warning {{incomplete universal character name; treating as '\' followed by identifier}}
M(\U123456)    // expected-warning {{incomplete universal character name; treating as '\' followed by identifier}}
M(\U1234567)   // expected-warning {{incomplete universal character name; treating as '\' followed by identifier}}
M(\U12345678)  // Okay
M(\U123456789) // Okay-ish, two tokens (valid-per-spec-but-actually-invalid UCN followed by 9)

// Now test the ones that should work. Note, these work in C17 and earlier but
// are part of the basic character set in C23 and thus should be diagnosed in
// that mode. They're valid in a character constant, but not valid in an
// identifier, except for U+0024 which is allowed if -fdollars-in-identifiers
// is enabled.
// FIXME: These three should be handled the same way, and should be accepted
// when dollar signs are allowed in identifiers, rather than rejected, see
// GH87106.
M(\u0024) // expected-error {{character '$' cannot be specified by a universal character name}}
M(\U00000024) // expected-error {{character '$' cannot be specified by a universal character name}}
M($)

// These should always be rejected because they're not valid identifier
// characters.
// FIXME: the diagnostic could be improved to make it clear this is an issue
// with forming an identifier rather than a UCN.
M(\u0040) // expected-error {{character '@' cannot be specified by a universal character name}}
M(\u0060) // expected-error {{character '`' cannot be specified by a universal character name}}
M(\U00000040) // expected-error {{character '@' cannot be specified by a universal character name}}
M(\U00000060) // expected-error {{character '`' cannot be specified by a universal character name}}

// UCNs outside of identifiers are handled in Phase 5 of translation, so we
// cannot use the macro expansion to test their behavior.

// This is outside of the range of values specified by ISO 10646.
const char *c1 = "\U00110000"; // expected-error {{invalid universal character}}
// This does not fall outside of the range
const char *c2 = "\U0010FFFF";

// These should always be accepted because they're a valid in a character
// constant.
int c3 = '\u0024';
int c4 = '\u0040';
int c5 = '\u0060';

int c6 = '\U00000024';
int c7 = '\U00000040';
int c8 = '\U00000060';

// Valid lone surrogates.
M(\uD799)
const char *c9 = "\U0000E000";

// Invalid lone surrogates, which are excluded explicitly by 6.4.3p2.
M(\uD800) // expected-error {{invalid universal character}}
const char *c10  = "\U0000DFFF"; // expected-error {{invalid universal character}}

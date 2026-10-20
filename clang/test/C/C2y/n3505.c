// RUN: %clang_cc1 -verify -std=c2y -Wall -pedantic %s
// RUN: %clang_cc1 -verify -std=c23 -Wall -pedantic %s
// RUN: %clang_cc1 -verify -std=c11 -Wall -pedantic %s
// RUN: %clang_cc1 -verify -std=c99 -Wall -pedantic %s

/* WG14 N3505: Yes
 * Preprocessor integer expressions, v. 2
 *
 * This introduces a constraint that preprocessing tokens must be an integer
 * literal, character literal, punctuator, or some other implementation-defined
 * sequence of tokens (to support builtins that insert odd tokens into the
 * parsing stream).
 */

// This is technically an integer constant expression, but it does not match
// the new constraints and thus needs to be diagnosed.
#if 1 ? 1 : (""[0] += 5) // expected-error {{invalid token at start of a preprocessor expression}}
#endif

// But with a character literal, it is fine.
#if 1 ? 1 : ('a' + 5) // Ok
#endif

// This doesn't mean that all punctuators are fine, however.
#if 1 ? 1 : ('a' += 5) // expected-error {{token is not a valid binary operator in a preprocessor subexpression}}
#endif

// But some are.
#if 1 ? 1 : ~('a') // Ok
#endif

int x; // Needs a declaration to avoid a pedantic warning

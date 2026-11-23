// RUN: %clang_cc1 -verify -std=c2y -Wall -pedantic %s
// RUN: %clang_cc1 -verify -std=c23 -Wall -pedantic %s

/* WG14 N3525: Yes
 * static_assert without UB
 *
 * Ensures that a static_assert declaration cannot defer to runtime; it must
 * take an integer constant expression that is resolved at compile time.
 *
 * Note: implementations are free to extend what is a valid integer constant
 * expression, and Clang (and GCC) does so. So this test is validating that
 * we quietly accept a pasing assertion, loudly reject a failing assertion, and
 * issue a pedantic diagnostic for the extension case.
 */

static_assert(1); // Okay

static_assert(0); // expected-error {{static assertion failed}}

extern int a;
static_assert(1 || a); // expected-warning {{expression is not an integer constant expression; folding it to a constant is a GNU extension}}

static_assert(a);      // expected-error {{static assertion expression is not an integral constant expression}}
static_assert(0 || a); // expected-error {{static assertion expression is not an integral constant expression}}

// Note, there is no CodeGen test for this; we have existing tests for the ICE
// extension, so the pedantic warning is sufficient to verify we're not
// emitting code which reads 'a' in '1 || a' because of the folding, and
// there's no way to generate code for reading 'a' in '0 || a' because of the
// error.

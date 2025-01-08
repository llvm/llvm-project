// RUN: %clang_cc1 -verify -std=c99 %s
// RUN: %clang_cc1 -E -std=c99 %s | FileCheck %s
// expected-no-diagnostics

/* WG14 N570: Yes
 * Empty macro arguments
 *
 * NB: the original paper is not available online anywhere, so the test
 * coverage is coming from what could be gleaned from the C99 rationale
 * document. In C89, it was UB to pass no arguments to a function-like macro,
 * and that's now supported in C99.
 */

#define TEN 10
#define U u
#define I // expands into no preprocessing tokens
#define L L
#define glue(a, b) a ## b
#define xglue(a, b) glue(a, b)

const unsigned u = xglue(TEN, U);
const int i = xglue(TEN, I);
const long l = xglue(TEN, L);

// CHECK: const unsigned u = 10u;
// CHECK-NEXT: const int i = 10;
// CHECK-NEXT: const long l = 10L;

_Static_assert(u == 10U, "");
_Static_assert(i == 10, "");
_Static_assert(l == 10L, "");

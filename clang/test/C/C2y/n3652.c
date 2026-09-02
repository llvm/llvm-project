// RUN: %clang_cc1 -verify -std=c2y -Wall -pedantic -Wno-unused %s

/* WG14 N3652: No
 * Composite types, v1.3
 *
 * For the conditional operator, constraints involving nullptr_t and pointers
 * to variably modified types are added.
 *
 * FIXME: Clang doesn't yet implement this paper.
 */

// expected-no-diagnostics

// FIXME: Should diagnose these.
void test(bool cond, void* p1, void* p2) {
  int n  = 2;
  auto a = cond ? nullptr : (char(*)[n])p1;
  auto b = cond ? (char(*)[])p1 : (char(*)[n])p2;
}

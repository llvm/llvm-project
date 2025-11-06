/* RUN: %clang_cc1 -verify -std=c99 -pedantic %s
   RUN: %clang_cc1 -verify=c89 -std=c89 -pedantic %s
   expected-no-diagnostics
 */

/* WG14 N782: Clang 3.4
 * Relaxed constraints on aggregate and union initialization
 */

void test(void) {
  struct S {
    int x, y;
  };
  int a = 1, b = 2;
  struct S s = { a, b }; /* c89-warning {{initializer for aggregate is not a compile-time constant}} */

  union U {
    int x;
    float f;
  };
  union U u = { a }; /* c89-warning {{initializer for aggregate is not a compile-time constant}} */
}


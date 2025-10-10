// RUN: %clang_cc1 -verify=good -pedantic -Wall -std=c2y %s
// RUN: %clang_cc1 -verify -pedantic -Wall -std=c23 %s
// RUN: %clang_cc1 -verify -pedantic -Wall -std=c17 %s
// good-no-diagnostics

/* WG14 N3622: Clang 22
 * Allow calling static inline within extern inline
 *
 * This verifies that a constraint from previous standards is no longer
 * triggered in C2y mode. The constraint is with calling a statric function
 * or using a static variable from an inline function with external linkage.
 */

static void static_func(void) {} // expected-note {{declared here}}
static int static_var;           // expected-note {{declared here}}

extern inline void test(void) {
  static_func();   // expected-warning {{static function 'static_func' is used in an inline function with external linkage}}
  static_var = 12; // expected-warning {{static variable 'static_var' is used in an inline function with external linkage}}
}

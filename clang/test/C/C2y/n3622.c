// RUN: %clang_cc1 -verify=good -pedantic -Wall -std=c2y %s
// RUN: %clang_cc1 -verify=compat,expected -pedantic -Wall -Wpre-c2y-compat -std=c2y %s
// RUN: %clang_cc1 -verify=ped,expected -pedantic -Wall -std=c23 %s
// RUN: %clang_cc1 -verify=ped,expected -pedantic -Wall -std=c17 %s
// good-no-diagnostics

/* WG14 N3622: Clang 22
 * Allow calling static inline within extern inline
 *
 * This verifies that a constraint from previous standards is no longer
 * triggered in C2y mode. The constraint is with calling a static function
 * or using a static variable from an inline function with external linkage.
 */

static void static_func(void) {} // expected-note {{declared here}}
static int static_var;           // expected-note {{declared here}}

extern inline void test(void) {
  static_func();   /* ped-warning {{using static function 'static_func' in an inline function with external linkage is a C2y extension}}
                      compat-warning {{using static function 'static_func' in an inline function with external linkage is incompatible with standards before C2y}}
                    */
  static_var = 12; /* ped-warning {{using static variable 'static_var' in an inline function with external linkage is a C2y extension}}
                      compat-warning {{using static variable 'static_var' in an inline function with external linkage is incompatible with standards before C2y}}
                    */
}

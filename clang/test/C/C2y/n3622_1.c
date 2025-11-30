// RUN: %clang_cc1 -verify=good -pedantic -Wall -std=c2y %s
// RUN: %clang_cc1 -verify=compat,expected -pedantic -Wall -Wpre-c2y-compat -std=c2y %s
// RUN: %clang_cc1 -verify=ped,expected -pedantic -Wall -std=c23 %s
// RUN: %clang_cc1 -verify=ped,expected -pedantic -Wall -std=c17 %s
// good-no-diagnostics

/* WG14 N3622: Clang 22
 * Allow static local variables in extern inline functions
 *
 * This verifies that a constraint from previous standards is no longer
 * triggered in C2y mode. The constraint is regarding static local
 * variables in inline functions with external linkage.
 */

inline void func1(void) {   // expected-note {{use 'static' to give inline function 'func1' internal linkage}}
    static int x = 0;   /* ped-warning {{non-constant static local variable in an inline function with external linkage is a C2y extension}}
                      compat-warning {{non-constant static local variable in an inline function with external linkage is incompatible with standards before C2y}}
                    */
    (void)x;
}

inline void func2(void) {
    static const int x = 0;
    (void)x;
}

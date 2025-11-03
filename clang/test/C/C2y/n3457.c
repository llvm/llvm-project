// RUN: %clang_cc1 -verify=ext -std=c23 -pedantic %s
// RUN: %clang_cc1 -verify=ext -pedantic -x c++ %s
// RUN: %clang_cc1 -verify=pre -std=c2y -pedantic -Wpre-c2y-compat %s

/* WG14 N3457: Clang 22
 * The __COUNTER__ predefined macro
 *
 * This predefined macro was supported as an extension in earlier versions of
 * Clang, but the required diagnostics for the limits were not added until 22.
 */

// Ensure that __COUNTER__ starts from 0.
static_assert(__COUNTER__ == 0); /* ext-warning {{'__COUNTER__' is a C2y extension}}
                                    pre-warning {{'__COUNTER__' is incompatible with standards before C2y}}
                                  */

// Ensure that the produced value can be used with token concatenation.
#define CAT_IMPL(a, b) a ## b
#define CAT(a, b) CAT_IMPL(a, b)
#define NAME_WITH_COUNTER(a) CAT(a, __COUNTER__)
void test() {
  // Because this is the 2nd expansion, this defines test1.
  int NAME_WITH_COUNTER(test); /* ext-warning {{'__COUNTER__' is a C2y extension}}
                                  pre-warning {{'__COUNTER__' is incompatible with standards before C2y}}
                                */
  int other_test = test1;      // Ok
}

// Ensure that __COUNTER__ increments each time you mention it.
static_assert(__COUNTER__ == 2); /* ext-warning {{'__COUNTER__' is a C2y extension}}
                                    pre-warning {{'__COUNTER__' is incompatible with standards before C2y}}
                                 */
static_assert(__COUNTER__ == 3); /* ext-warning {{'__COUNTER__' is a C2y extension}}
                                    pre-warning {{'__COUNTER__' is incompatible with standards before C2y}}
                                 */
static_assert(__COUNTER__ == 4); /* ext-warning {{'__COUNTER__' is a C2y extension}}
                                    pre-warning {{'__COUNTER__' is incompatible with standards before C2y}}
                                 */

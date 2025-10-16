// RUN: %clang_cc1 -verify -std=c23 %s
// RUN: %clang_cc1 -verify=pedantic -std=c11 -pedantic %s
// RUN: %clang_cc1 -verify=compat -std=c23 -Wpre-c23-compat %s

// expected-no-diagnostics

/* WG14 N2508: yes
 * Free positioning of labels inside compound statements
 */
void test(void) {
  {
  inner:
  } /* pedantic-warning {{label at end of compound statement is a C23 extension}}
       compat-warning {{label at end of compound statement is incompatible with C standards before C23}}
     */

  switch (1) {
  case 1:
  } /* pedantic-warning {{label at end of compound statement is a C23 extension}}
       compat-warning {{label at end of compound statement is incompatible with C standards before C23}}
     */

  {
  multiple: labels: on: a: line:
  } /* pedantic-warning {{label at end of compound statement is a C23 extension}}
       compat-warning {{label at end of compound statement is incompatible with C standards before C23}}
     */

final:
} /* pedantic-warning {{label at end of compound statement is a C23 extension}}
     compat-warning {{label at end of compound statement is incompatible with C standards before C23}}
   */

void test_labels(void) {
label:
  int i = 0; /* pedantic-warning {{label followed by a declaration is a C23 extension}}
                compat-warning {{label followed by a declaration is incompatible with C standards before C23}}
              */

  switch (i) {
  case 1:
    _Static_assert(1, ""); /* pedantic-warning {{label followed by a declaration is a C23 extension}}
                              compat-warning {{label followed by a declaration is incompatible with C standards before C23}}
                            */
  default:
    _Static_assert(1, ""); /* pedantic-warning {{label followed by a declaration is a C23 extension}}
                              compat-warning {{label followed by a declaration is incompatible with C standards before C23}}
                            */
  }
}

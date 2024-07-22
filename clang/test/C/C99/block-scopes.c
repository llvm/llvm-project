// RUN: %clang_cc1 -std=c89 -verify %s
// RUN: %clang_cc1 -std=c99 -verify %s
// RUN: %clang_cc1 -std=c11 -verify %s
// RUN: %clang_cc1 -std=c17 -verify %s
// RUN: %clang_cc1 -std=c23 -verify %s

// expected-no-diagnostics

/* WG14 ???: yes
 * new block scopes for selection and iteration statements
 *
 * This is referenced in the C99 front matter as new changes to C99, but it is
 * not clear which document number introduced the changes. It's possible this
 * is WG14 N759, based on discussion in the C99 rationale document that claims
 * these changes were made in response to surprising issues with the lifetime
 * of compound literals in compound statements vs non-compound statements.
 */

enum {a, b};
void different(void) {
  if (sizeof(enum {b, a}) != sizeof(int)) {
    _Static_assert(a == 1, "");
  }
  /* In C89, the 'b' found here would have been from the enum declaration in
   * the controlling expression of the selection statement, not from the global
   * declaration. In C99 and later, that enumeration is scoped to the 'if'
   * statement and the global declaration is what's found.
   */
  #if __STDC_VERSION__ >= 199901L
    _Static_assert(b == 1, "");
  #else
    _Static_assert(b == 0, "");
  #endif
}


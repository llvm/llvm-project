/* RUN: %clang_cc1 -std=c89 -fsyntax-only -pedantic -verify %s
   RUN: %clang_cc1 -std=c99 -fsyntax-only -pedantic -verify %s
   RUN: %clang_cc1 -std=c11 -fsyntax-only -pedantic -verify %s
   RUN: %clang_cc1 -std=c17 -fsyntax-only -pedantic -verify %s
   RUN: %clang_cc1 -std=c2x -fsyntax-only -pedantic -verify %s
 */

/* WG14 DR157: yes
 * Legitimacy of type synonyms
 *
 * Part 1 is about whether you can use a typedef to void in place of void in
 * a function parameter list and still get a function with a prototype that
 * accepts no arguments. You can.
 *
 * Part 2 is about whether you can use a typedef to int in place of int in
 * the declaration of main(). You can.
 *
 * Part 3 is about whether there are situations where a typedef cannot be used
 * in place of a type name.
 */
typedef void dr157_1_t;
extern int dr157(dr157_1_t); /* ok */
int dr157(dr157_1_t) { /* ok */
  /* You cannot combine a typedef with another type specifier. */
  typedef int Int; /* expected-note {{previous definition is here}} */
  long Int val;    /* expected-error {{redefinition of 'Int' as different kind of symbol}}
                      expected-error {{expected ';' at end of declaration}}
                   */

  return 0;
}

typedef int dr157_2_t;
dr157_2_t main(void) { /* Still a valid declaration of main() */
}

/* A function definition cannot use a typedef for the type. */
typedef void dr157_3_t(void);
extern dr157_3_t dr157_2 { /* expected-error {{expected ';' after top level declarator}} */
}

/* FIXME: all diagnostics that happen after the previous one about expecting a
 * a ';' are silenced, so this test needs to be in its own file to prevent
 * accidentally incorrect testing.
 */

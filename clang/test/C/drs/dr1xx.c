/* RUN: %clang_cc1 -std=c89 -verify=expected,c89only -pedantic -Wno-c11-extensions %s
   RUN: %clang_cc1 -std=c99 -verify=expected,c99untilc2x -pedantic -Wno-c11-extensions %s
   RUN: %clang_cc1 -std=c11 -verify=expected,c99untilc2x -pedantic %s
   RUN: %clang_cc1 -std=c17 -verify=expected,c99untilc2x -pedantic %s
   RUN: %clang_cc1 -std=c2x -verify=expected,c2xandup -pedantic %s
 */

/* The following are DRs which do not require tests to demonstrate
 * conformance or nonconformance.
 *
 * WG14 DR100: dup 001
 * Defect with the return statement
 *
 * WG14 DR104: dup 084
 * Incomplete tag types in a parameter list
 */


/* WG14 DR101: yes
 * Type qualifiers and "as if by assignment"
 */
void dr101_callee(const int val);
void dr101_caller(void) {
  int val = 1;
  dr101_callee(val); /* ok; const qualifier on the parameter doesn't prevent as-if assignment. */
}

/* WG14 DR102: yes
 * Tag redeclaration constraints
 */
void dr102(void) {
  struct S { int member; }; /* expected-note {{previous definition is here}} */
  struct S { int member; }; /* expected-error {{redefinition of 'S'}} */

  union U { int member; }; /* expected-note {{previous definition is here}} */
  union U { int member; }; /* expected-error {{redefinition of 'U'}} */

  enum E { member }; /* expected-note 2{{previous definition is here}} */
  enum E { member }; /* expected-error {{redefinition of 'E'}}
                        expected-error {{redefinition of enumerator 'member'}} */
}

/* WG14 DR103: yes
 * Formal parameters of incomplete type
 */
void dr103_1(int arg[]); /* ok, not an incomplete type due to rewrite */
void dr103_2(struct S s) {} /* expected-warning {{declaration of 'struct S' will not be visible outside of this function}}
                               expected-error {{variable has incomplete type 'struct S'}}
                               expected-note {{forward declaration of 'struct S'}} */
void dr103_3(struct S s);               /* expected-warning {{declaration of 'struct S' will not be visible outside of this function}}
                                           expected-note {{previous declaration is here}} */
void dr103_3(struct S { int a; } s) { } /* expected-warning {{declaration of 'struct S' will not be visible outside of this function}}
                                           expected-error {{conflicting types for 'dr103_3'}} */
void dr103_4(struct S s1, struct S { int a; } s2); /* expected-warning {{declaration of 'struct S' will not be visible outside of this function}} */

/* WG14 DR105: dup 017
 * Precedence of requirements on compatible types
 *
 * NB: This is also Question 3 from DR017.
 */
void dr105(void) {
  /* According to C2x 6.7.6.3p14 the return type and parameter types to be
   * compatible types, but qualifiers are dropped from the parameter type.
   */
  extern void func(int);
  extern void func(const int); /* FIXME: this should be pedantically diagnosed. */

  extern void other_func(int);   /* expected-note {{previous declaration is here}} */
  extern void other_func(int *); /* expected-error {{conflicting types for 'other_func'}} */

  extern int i;   /* expected-note {{previous declaration is here}} */
  extern float i; /* expected-error {{redeclaration of 'i' with a different type: 'float' vs 'int'}} */
}

// RUN: %clang_cc1 -verify -std=c2y %s
// RUN: %clang_cc1 -verify -std=c23 %s
// RUN: %clang_cc1 -verify -std=c17 %s
// RUN: %clang_cc1 -verify -std=c11 %s
// RUN: %clang_cc1 -verify -std=c99 %s
// RUN: %clang_cc1 -verify -std=c89 %s

/* WG14 N3532: Yes
 * Member access of an incomplete object
 *
 * Verify that the first operand to the . or -> operators is a complete object
 * type.
 */

struct S {
  int i;
};

union U {
  int i;
};

void good_test(void) {
  struct S s;
  struct S *s_ptr = &s;
  union U u;
  union U *u_ptr = &u;

  // Complete object type, correctly named member.
  s.i = 10;
  s_ptr->i = 10;
  u.i = 10;
  u_ptr->i = 10;
}

void bad_test(void) {
  struct Incomplete *s_ptr;    /* expected-note 2 {{forward declaration of 'struct Incomplete'}} */
  union AlsoIncomplete *u_ptr; /* expected-note 2 {{forward declaration of 'union AlsoIncomplete'}} */
  struct S s;
  union U u;

  // Incomplete object type.
  s_ptr->i = 10; /* expected-error {{incomplete definition of type 'struct Incomplete'}} */
  u_ptr->i = 10; /* expected-error {{incomplete definition of type 'union AlsoIncomplete'}} */

  (*s_ptr).i = 10; /* expected-error {{incomplete definition of type 'struct Incomplete'}} */
  (*u_ptr).i = 10; /* expected-error {{incomplete definition of type 'union AlsoIncomplete'}} */

  // Complete object type, no named member.
  s.f = "test"; /* expected-error {{no member named 'f' in 'struct S'}} */
  u.f = "test"; /* expected-error {{no member named 'f' in 'union U'}} */
}


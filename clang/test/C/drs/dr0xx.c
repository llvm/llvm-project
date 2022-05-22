/* RUN: %clang_cc1 -std=c89 -verify=expected,c89 -pedantic -Wno-declaration-after-statement -Wno-c11-extensions %s
   RUN: %clang_cc1 -std=c99 -verify -pedantic -Wno-c11-extensions %s
   RUN: %clang_cc1 -std=c11 -verify -pedantic %s
   RUN: %clang_cc1 -std=c17 -verify -pedantic %s
   RUN: %clang_cc1 -std=c2x -verify -pedantic %s
 */

/* The following are DRs which do not require tests to demonstrate
 * conformance or nonconformance.
 *
 * WG14 DR005: yes
 * May a conforming implementation define and recognize a pragma which would
 * change the semantics of the language?
 *
 * WG14 DR008: yes
 * Can a conforming C compiler to perform dead-store elimination?
 *
 * WG14 DR020: yes
 * Is a compiler which allows the Relaxed Ref/Def linkage model to be
 * considered a conforming compiler?
 *
 * WG14 DR025: yes
 * What is meant by 'representable floating-point value?'
 *
 * WG14 DR026: yes
 * Can a strictly conforming program contain a string literal with '$' or '@'?
 *
 * WG14 DR033: yes
 * Conformance questions around 'shall' violations outside of constraints
 * sections
 */


/* WG14 DR004: yes
 * Are multiple definitions of unused identifiers with external linkage
 * permitted?
 */
int dr004(void) {return 0;} /* expected-note {{previous definition is here}} */
int dr004(void) {return 1;} /* expected-error {{redefinition of 'dr004'}} */

/* WG14 DR007: yes
 * Are declarations of the form struct-or-union identifier ; permitted after
 * the identifier tag has already been declared?
 */
struct dr007_a;
struct dr007_a;
struct dr007_a {int a;};
struct dr007_a;
struct dr007_b {int a;};
struct dr007_b;

/* WG14 DR009: no
 * Use of typedef names in parameter declarations
 */
typedef int dr009_t;
void dr009_f(int dr009_t);

/* WG14 DR010:
 * Is a typedef to an incomplete type legal?
 */
typedef int dr010_t[];
dr010_t dr010_a = {1};
dr010_t dr010_b = {1, 2};
int dr010_c = sizeof(dr010_t); /* expected-error {{invalid application of 'sizeof' to an incomplete type 'dr010_t' (aka 'int[]')}} */

/* WG14 DR011: yes
 * Merging of declarations for linked identifier
 *
 * Note: more of this DR is tested in dr011.c
 */
static int dr011_a[]; /* expected-warning {{tentative array definition assumed to have one element}} */
void dr011(void) {
  extern int i[];
  {
    /* a different declaration of the same object */
    extern int i[10];
    (void)sizeof(i);
    _Static_assert(sizeof(i) == 10 * sizeof(int), "fail");
  }
  (void)sizeof(i); /* expected-error {{invalid application of 'sizeof' to an incomplete type 'int[]'}} */

  extern int dr011_a[10];
  (void)sizeof(dr011_a);
  _Static_assert(sizeof(dr011_a) == 10 * sizeof(int), "fail");

  extern int j[10];
  {
    extern int j[];
    (void)sizeof(j);
    _Static_assert(sizeof(j) == 10 * sizeof(int), "fail");
  }
}

/* WG14 DR012: yes
 * Is it valid to take the address of a dereferenced void pointer?
 */
void dr012(void *p) {
  /* The behavior changed between C89 and C99. */
  (void)&*p; /* c89-warning {{ISO C forbids taking the address of an expression of type 'void'}} */
}

/* WG14 DR013: yes
 * Compatible and composite function types
 */
int dr013(int a[4]);
int dr013(int a[5]);
int dr013(int *a);

struct dr013_t {
struct dr013_t *p;
} dr013_v[sizeof(struct dr013_t)];

/* WG14 DR015: yes
 * What is the promoted type of a plain int bit-field?
 */
void dr015(void) {
  struct S {
    int small_int_bitfield : 16;
    unsigned int small_uint_bitfield : 16;
    int int_bitfield : 32;
    unsigned int uint_bitfield : 32;
  } s;
  _Static_assert(__builtin_types_compatible_p(__typeof__(+s.small_int_bitfield), int), "fail");
  _Static_assert(__builtin_types_compatible_p(__typeof__(+s.small_uint_bitfield), int), "fail");
  _Static_assert(__builtin_types_compatible_p(__typeof__(+s.int_bitfield), int), "fail");
  _Static_assert(__builtin_types_compatible_p(__typeof__(+s.uint_bitfield), unsigned int), "fail");
}

/* WG14 DR027: yes
 * Can there be characters in the character set that are not in the required
 * source character set?
 */
#define THIS$AND$THAT(a, b) ((a) + (b)) /* expected-warning 2 {{'$' in identifier}} */
_Static_assert(THIS$AND$THAT(1, 1) == 2, "fail"); /* expected-warning 2 {{'$' in identifier}} */


/* WG14 DR029: no
 * Do two types have to have the same tag to be compatible?
 * Note: the rule changed in C99 to be different than the resolution to DR029,
 * so it's not clear there's value in implementing this DR.
 */
_Static_assert(__builtin_types_compatible_p(struct S { int a; }, union U { int a; }), "fail"); /* expected-error {{static_assert failed due to requirement '__builtin_types_compatible_p(struct S, union U)' "fail"}} */

/* WG14 DR031: yes
 * Can constant expressions overflow?
 */
void dr031(int i) {
  switch (i) {
  case __INT_MAX__ + 1: break; /* expected-warning {{overflow in expression; result is -2147483648 with type 'int'}} */
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wswitch"
  /* Silence the targets which issue:
   * warning: overflow converting case value to switch condition type (2147483649 to 18446744071562067969)
   */
  case __INT_MAX__ + 2ul: break;
  #pragma clang diagnostic pop
  case (__INT_MAX__ * 4) / 4: break; /* expected-warning {{overflow in expression; result is -4 with type 'int'}} */
  }
}

/* WG21 DR032: no
 * Must implementations diagnose extensions to the constant evaluation rules?
 *
 * This should issue a diagnostic because a constant-expression is a
 * conditional-expression, which excludes the comma operator.
 */
int dr032 = (1, 2); /* expected-warning {{left operand of comma operator has no effect}} */

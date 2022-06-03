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
 *
 * WG14 DR109: yes
 * Are undefined values and undefined behavior the same?
 *
 * WG14 DR110: dup 047
 * Formal parameters having array-of-non-object types
 *
 * WG14 DR117: yes
 * Abstract semantics, sequence points, and expression evaluation
 *
 * WG14 DR121: yes
 * Conversions of pointer values to integral types
 *
 * WG14 DR122: dup 015
 * Conversion/widening of bit-fields
 *
 * WG14 DR125: yes
 * Using things declared as 'extern (qualified) void'
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

/* WG14 DR106: yes
 * When can you dereference a void pointer?
 *
 * NB: This is a partial duplicate of DR012.
 */
void dr106(void *p, int i) {
  /* The behavior changed between C89 and C99. */
  (void)&*p; /* c89only-warning {{ISO C forbids taking the address of an expression of type 'void'}} */
  /* The behavior of all three of these is undefined. */
  (void)*p;
  (void)(i ? *p : *p);
  (void)(*p, *p); /* expected-warning {{left operand of comma operator has no effect}} */
}

/* WG14 DR108: yes
 * Can a macro identifier hide a keyword?
 */
void dr108(void) {
#define const
  const int i = 12;
#undef const
  const int j = 12; /* expected-note {{variable 'j' declared const here}} */

  i = 100; /* Okay, the keyword was hidden by the macro. */
  j = 100; /* expected-error {{cannot assign to variable 'j' with const-qualified type 'const int'}} */
}

/* WG14 DR111: yes
 * Conversion of pointer-to-qualified type values to type (void*) values
 */
void dr111(const char *ccp, void *vp) {
  vp = ccp; /* expected-warning {{assigning to 'void *' from 'const char *' discards qualifiers}} */
}

/* WG14 DR112: yes
 * Null pointer constants and relational comparisons
 */
void dr112(void *vp) {
  /* The behavior of this expression is pedantically undefined.
   * FIXME: should we diagnose under -pedantic?
   */
  (void)(vp > (void*)0);
}

/* WG14 DR113: yes
 * Return expressions in functions declared to return qualified void
 */
volatile void dr113_v(volatile void *vvp) { /* expected-warning {{function cannot return qualified void type 'volatile void'}} */
  return *vvp; /* expected-warning {{void function 'dr113_v' should not return void expression}} */
}
const void dr113_c(const void *cvp) { /* expected-warning {{function cannot return qualified void type 'const void'}} */
  return *cvp; /* expected-warning {{void function 'dr113_c' should not return void expression}} */
}

/* WG14 DR114: yes
 * Initialization of multi-dimensional char array objects
 */
void dr114(void) {
  char array[2][5] = { "defghi" }; /* expected-warning {{initializer-string for char array is too long}} */
}

/* WG14 DR115: yes
 * Member declarators as declarators
 */
void dr115(void) {
  struct { int mbr; }; /* expected-warning {{declaration does not declare anything}} */
  union { int mbr; };  /* expected-warning {{declaration does not declare anything}} */
}

/* WG14 DR116: yes
 * Implicit unary & applied to register arrays
 */
void dr116(void) {
  register int array[5] = { 0, 1, 2, 3, 4 };
  (void)array;       /* expected-error {{address of register variable requested}} */
  (void)array[3];    /* expected-error {{address of register variable requested}} */
  (void)(array + 3); /* expected-error {{address of register variable requested}} */
}

/* WG14 DR118: yes
 * Completion point for enumerated types
 */
void dr118(void) {
  enum E {
	/* The enum isn't a complete type until the closing }, but an
	 * implementation may complete the type earlier if it has sufficient type
	 * information to calculate size or alignment, etc.
	 */
    Val = sizeof(enum E)
  };
}

/* WG14 DR119: yes
 * Initialization of multi-dimensional array objects
 */
void dr119(void) {
  static int array[][] = { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 } }; /* expected-error {{array has incomplete element type 'int[]'}} */
}

/* WG14 DR120: yes
 * Semantics of assignment to (and initialization of) bit-fields
 */
void dr120(void) {
  /* We could verify this one with a codegen test to ensure that the proper
   * value is stored into bit, but the diagnostic tells us what the value is
   * after conversion, so we can lean on that for verification.
   */
  struct S { unsigned bit:1; };
  struct S object1 = { 3 }; /* expected-warning {{implicit truncation from 'int' to bit-field changes value from 3 to 1}} */
  struct S object2;
  object2.bit = 3; /* expected-warning {{implicit truncation from 'int' to bit-field changes value from 3 to 1}} */
}

/* WG14 DR123: yes
 * 'Type categories' and qualified types
 */
void dr123(void) {
  /* Both of these examples are strictly conforming. */
  enum E1 {
    enumerator1 = (const int) 9
  };
  enum E2 {
    enumerator2 = (volatile int) 9
  };
}

/* WG14 DR124: yes
 * Casts to 'a void type' versus casts to 'the void type'
 */
void dr124(void) {
  /* A cast can cast to void or any qualified version of void. */
  (const volatile void)0;
}

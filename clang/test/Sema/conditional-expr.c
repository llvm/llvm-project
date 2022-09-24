// RUN: %clang_cc1 -fsyntax-only -Wno-pointer-to-int-cast -verify -pedantic -Wsign-conversion %s
void foo(void) {
  *(0 ? (double *)0 : (void *)0) = 0;
  // FIXME: GCC doesn't consider the following two statements to be errors.
  *(0 ? (double *)0 : (void *)(int *)0) = 0; /* expected-error {{incomplete type 'void' is not assignable}}
                                                expected-warning {{ISO C does not allow indirection on operand of type 'void *'}} */
  *(0 ? (double *)0 : (void *)(double *)0) = 0; /* expected-error {{incomplete type 'void' is not assignable}}
                                                   expected-warning {{ISO C does not allow indirection on operand of type 'void *'}} */
  *(0 ? (double *)0 : (int *)(void *)0) = 0; /* expected-error {{incomplete type 'void' is not assignable}}
                                                expected-warning {{pointer type mismatch ('double *' and 'int *')}}
                                                expected-warning {{ISO C does not allow indirection on operand of type 'void *'}} */
  *(0 ? (double *)0 : (double *)(void *)0) = 0;
  *((void *) 0) = 0; /* expected-error {{incomplete type 'void' is not assignable}}
                        expected-warning {{ISO C does not allow indirection on operand of type 'void *'}} */
  double *dp;
  int *ip;
  void *vp;

  dp = vp;
  vp = dp;
  ip = dp; // expected-warning {{incompatible pointer types assigning to 'int *' from 'double *'}}
  dp = ip; // expected-warning {{incompatible pointer types assigning to 'double *' from 'int *'}}
  dp = 0 ? (double *)0 : (void *)0;
  vp = 0 ? (double *)0 : (void *)0;
  ip = 0 ? (double *)0 : (void *)0; // expected-warning {{incompatible pointer types assigning to 'int *' from 'double *'}}

  const int *cip;
  vp = (0 ? vp : cip); // expected-warning {{discards qualifiers}}
  vp = (0 ? cip : vp); // expected-warning {{discards qualifiers}}

  int i = 2;
  int (*pf)[2];
  int (*pv)[i];
  pf = (i ? pf : pv);

  enum {xxx,yyy,zzz} e, *ee;
  short x;
  ee = ee ? &x : ee ? &i : &e; // expected-warning {{pointer type mismatch}}

  typedef void *asdf;
  *(0 ? (asdf) 0 : &x) = 10;

  unsigned long test0 = 5;
  test0 = test0 ? (long) test0 : test0; // expected-warning {{operand of ? changes signedness: 'long' to 'unsigned long'}}
  test0 = test0 ? (int) test0 : test0; // expected-warning {{operand of ? changes signedness: 'int' to 'unsigned long'}}
  test0 = test0 ? (short) test0 : test0; // expected-warning {{operand of ? changes signedness: 'short' to 'unsigned long'}}
  test0 = test0 ? test0 : (long) test0; // expected-warning {{operand of ? changes signedness: 'long' to 'unsigned long'}}
  test0 = test0 ? test0 : (int) test0; // expected-warning {{operand of ? changes signedness: 'int' to 'unsigned long'}}
  test0 = test0 ? test0 : (short) test0; // expected-warning {{operand of ? changes signedness: 'short' to 'unsigned long'}}
  test0 = test0 ? test0 : (long) 10;
  test0 = test0 ? test0 : (int) 10;
  test0 = test0 ? test0 : (short) 10;
  test0 = test0 ? (long) 10 : test0;
  test0 = test0 ? (int) 10 : test0;
  test0 = test0 ? (short) 10 : test0;

  int test1;
  enum Enum { EVal };
  test0 = test0 ? EVal : test0;
  test1 = test0 ? EVal : (int) test0;
  test0 = test0 ?
                  (unsigned) EVal
                : (int) test0;  // expected-warning {{operand of ? changes signedness: 'int' to 'unsigned long'}}

  test0 = test0 ? EVal : test1; // expected-warning {{operand of ? changes signedness: 'int' to 'unsigned long'}}
  test0 = test0 ? test1 : EVal; // expected-warning {{operand of ? changes signedness: 'int' to 'unsigned long'}}

  const int *const_int;
  int *nonconst_int;
  *(test0 ? const_int : nonconst_int) = 42; // expected-error {{read-only variable is not assignable}}
  *(test0 ? nonconst_int : const_int) = 42; // expected-error {{read-only variable is not assignable}}

  // The composite type here should be "int (*)[12]", fine for the sizeof
  int (*incomplete)[];
  int (*complete)[12];
  sizeof(*(test0 ? incomplete : complete)); // expected-warning {{expression result unused}}
  sizeof(*(test0 ? complete : incomplete)); // expected-warning {{expression result unused}}

  int __attribute__((address_space(2))) *adr2;
  int __attribute__((address_space(3))) *adr3;
  test0 ? adr2 : adr3; // expected-error{{conditional operator with the second and third operands of type  ('__attribute__((address_space(2))) int *' and '__attribute__((address_space(3))) int *') which are pointers to non-overlapping address spaces}}

  // Make sure address-space mask ends up in the result type
  (test0 ? (test0 ? adr2 : adr2) : nonconst_int); // expected-error{{conditional operator with the second and third operands of type  ('__attribute__((address_space(2))) int *' and 'int *') which are pointers to non-overlapping address spaces}}
}

int Postgresql(void) {
  char x;
  return ((((&x) != ((void *) 0)) ? (*(&x) = ((char) 1)) : (void) ((void *) 0)), (unsigned long) ((void *) 0)); // expected-warning {{C99 forbids conditional expressions with only one void side}}
}

#define nil ((void*) 0)

extern int f1(void);

int f0(int a) {
  // GCC considers this a warning.
  return a ? f1() : nil; // expected-warning {{pointer/integer type mismatch in conditional expression ('int' and 'void *')}} expected-error {{incompatible pointer to integer conversion returning 'void *' from a function with result type 'int'}}
}

int f2(int x) {
  // We can suppress this because the immediate context wants an int.
  return (x != 0) ? 0U : x;
}

#define NULL (void*)0

void PR9236(void) {
  struct A {int i;} A1;
  (void)(1 ? A1 : NULL); // expected-error{{non-pointer operand type 'struct A' incompatible with NULL}}
  (void)(1 ? NULL : A1); // expected-error{{non-pointer operand type 'struct A' incompatible with NULL}}
  (void)(1 ? 0 : A1); // expected-error{{incompatible operand types}}
  (void)(1 ? (void*)0 : A1); // expected-error{{incompatible operand types}}
  (void)(1 ? A1: (void*)0); // expected-error{{incompatible operand types}}
  (void)(1 ? A1 : (NULL)); // expected-error{{non-pointer operand type 'struct A' incompatible with NULL}}
}


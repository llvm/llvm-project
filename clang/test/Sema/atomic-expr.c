// RUN: %clang_cc1 %s -verify=expected,access -fsyntax-only
// RUN: %clang_cc1 %s -std=c2x -verify=expected,access -fsyntax-only
// RUN: %clang_cc1 %s -std=c2x -pedantic -verify=expected,access -fsyntax-only
// RUN: %clang_cc1 %s -verify -fsyntax-only -Wno-atomic-access
// RUN: %clang_cc1 %s -verify=expected,access -fsyntax-only -fexperimental-new-constant-interpreter
// RUN: %clang_cc1 %s -std=c2x -verify=expected,access -fsyntax-only -fexperimental-new-constant-interpreter
// RUN: %clang_cc1 %s -std=c2x -pedantic -verify=expected,access -fsyntax-only -fexperimental-new-constant-interpreter
// RUN: %clang_cc1 %s -verify -fsyntax-only -Wno-atomic-access -fexperimental-new-constant-interpreter


_Atomic(unsigned int) data1;
int _Atomic data2;

// Shift operations

int func_01 (int x) {
  return data1 << x;
}

int func_02 (int x) {
  return x << data1;
}

int func_03 (int x) {
  return data2 << x;
}

int func_04 (int x) {
  return x << data2;
}

int func_05 (void) {
  return data2 << data1;
}

int func_06 (void) {
  return data1 << data2;
}

void func_07 (int x) {
  data1 <<= x;
}

void func_08 (int x) {
  data2 <<= x;
}

void func_09 (int* xp) {
  *xp <<= data1;
}

void func_10 (int* xp) {
  *xp <<= data2;
}

int func_11 (int x) {
  return data1 == x;
}

int func_12 (void) {
  return data1 < data2;
}

int func_13 (int x, unsigned y) {
  return x ? data1 : y;
}

int func_14 (void) {
  return data1 == 0;
}

void func_15(void) {
  // Ensure that the result of an assignment expression properly strips the
  // _Atomic qualifier; Issue 48742.
  _Atomic int x;
  int y = (x = 2);
  int z = (int)(x = 2);
  y = (x = 2);
  z = (int)(x = 2);
  y = (x += 2);

  _Static_assert(__builtin_types_compatible_p(__typeof__(x = 2), int), "incorrect");
  _Static_assert(__builtin_types_compatible_p(__typeof__(x += 2), int), "incorrect");
}

// Ensure that member access of an atomic structure or union type is properly
// diagnosed as being undefined behavior; Issue 54563.
void func_16(void) {
  // LHS member access.
  _Atomic struct { int val; } x, *xp;
  x.val = 12;   // access-error {{accessing a member of an atomic structure or union is undefined behavior}}
  xp->val = 12; // access-error {{accessing a member of an atomic structure or union is undefined behavior}}

  _Atomic union {
    int ival;
    float fval;
  } y, *yp;
  y.ival = 12;     // access-error {{accessing a member of an atomic structure or union is undefined behavior}}
  yp->fval = 1.2f; // access-error {{accessing a member of an atomic structure or union is undefined behavior}}

  // RHS member access.
  int xval = x.val; // access-error {{accessing a member of an atomic structure or union is undefined behavior}}
  xval = xp->val;   // access-error {{accessing a member of an atomic structure or union is undefined behavior}}
  int yval = y.ival; // access-error {{accessing a member of an atomic structure or union is undefined behavior}}
  yval = yp->ival;   // access-error {{accessing a member of an atomic structure or union is undefined behavior}}

  // Using the type specifier instead of the type qualifier.
  _Atomic(struct { int val; }) z;
  z.val = 12;       // access-error {{accessing a member of an atomic structure or union is undefined behavior}}
  int zval = z.val; // access-error {{accessing a member of an atomic structure or union is undefined behavior}}

  // Don't diagnose in an unevaluated context, however.
  (void)sizeof(x.val);
  (void)sizeof(xp->val);
  (void)sizeof(y.ival);
  (void)sizeof(yp->ival);
}

// Ensure that we correctly implement assignment constraints from C2x 6.5.16.1.
void func_17(void) {
  // The left operand has atomic ... arithmetic type, and the right operand has
  // arithmetic type;
  _Atomic int i = 0;
  _Atomic float f = 0.0f;

  // the left operand has an atomic ... version of a structure or union type
  // compatible with the type of the right operand;
  struct S { int i; } non_atomic_s;
  _Atomic struct S s = non_atomic_s;

  union U { int i; float f; } non_atomic_u;
  _Atomic union U u = non_atomic_u;

  // the left operand has atomic ... pointer type, and (considering the type
  // the left operand would have after lvalue conversion) both operands are
  // pointers to qualified or unqualified versions of compatible types, and the
  // type pointed to by the left operand has all the qualifiers of the type
  // pointed to by the right operand;
  const int *cip = 0;
  volatile const int *vcip = 0;
  const int * const cicp = 0;
  _Atomic(const int *) acip = cip;
  _Atomic(const int *) bad_acip = vcip; // expected-warning {{initializing '_Atomic(const int *)' with an expression of type 'const volatile int *' discards qualifiers}}
  _Atomic(const int *) acip2 = cicp;
  _Atomic(int *) aip = &i; // expected-warning {{incompatible pointer types initializing '_Atomic(int *)' with an expression of type '_Atomic(int) *'}} \

  // the left operand has atomic ... pointer type, and (considering the type
  // the left operand would have after lvalue conversion) one operand is a
  // pointer to an object type, and the other is a pointer to a qualified or
  // unqualified version of void, and the type pointed to by the left operand
  // has all the qualifiers of the type pointed to by the right operand;
  const void *cvp = 0;
  _Atomic(const int *) acip3 = cvp;
  _Atomic(const void *) acvip = cip;
  _Atomic(const int *) acip4 = vcip;   // expected-warning {{initializing '_Atomic(const int *)' with an expression of type 'const volatile int *' discards qualifiers}}
  _Atomic(const void *) acvip2 = vcip; // expected-warning {{initializing '_Atomic(const void *)' with an expression of type 'const volatile int *' discards qualifiers}}
  _Atomic(const int *) acip5 = cicp;
  _Atomic(const void *) acvip3 = cicp;

#if __STDC_VERSION__ >= 202311L
  // the left operand has an atomic ... version of the nullptr_t type and the
  // right operand is a null pointer constant or its type is nullptr_t
  typedef typeof(nullptr) nullptr_t;
  nullptr_t n;
  _Atomic nullptr_t cn2 = n;
  _Atomic nullptr_t cn3 = nullptr;
#endif // __STDC_VERSION__ >= 202311L

  // the left operand is an atomic ... pointer, and the right operand is a null
  // pointer constant or its type is nullptr_t;
  _Atomic(int *) aip2 = 0;
#if __STDC_VERSION__ >= 202311L
  _Atomic(int *) ip2 = n;
  _Atomic(int *) ip3 = nullptr;
  _Atomic(const int *) ip4 = nullptr;
#endif // __STDC_VERSION__ >= 202311L
}

// Ensure that the assignment constraints also work at file scope.
_Atomic int ai = 0;
_Atomic float af = 0.0f;
_Atomic(int *) aip1 = 0;

struct S { int a; } non_atomic_s;
_Atomic struct S as = non_atomic_s; // expected-error {{initializer element is not a compile-time constant}}

const int *cip = 0;
_Atomic(const int *) acip1 = cip; // expected-error {{initializer element is not a compile-time constant}}

const void *cvp = 0;
_Atomic(const int *) acip2 = cvp; // expected-error {{initializer element is not a compile-time constant}}

#if __STDC_VERSION__ >= 202311L
  // the left operand has an atomic ... version of the nullptr_t type and the
  // right operand is a null pointer constant or its type is nullptr_t
  typedef typeof(nullptr) nullptr_t;
  nullptr_t n;
  _Atomic nullptr_t cn2 = n; // expected-error {{initializer element is not a compile-time constant}}
  _Atomic(int *) aip2 = nullptr;
#endif // __STDC_VERSION__ >= 202311L

// FIXME: &ai is an address constant, so this should be accepted as an
// initializer, but the bit-cast inserted due to the pointer conversion is
// tripping up the test for whether the initializer is a constant expression.
// The warning is correct but the error is not.
_Atomic(int *) aip3 = &ai; /* expected-warning {{incompatible pointer types initializing '_Atomic(int *)' with an expression of type '_Atomic(int) *'}}
                              expected-error {{initializer element is not a compile-time constant}}
                            */

// Test the behavior when converting the null pointer constant to an atomic
// function pointer.
_Atomic(int (*)(char)) afp = (void *)0;

void func_18(void) {
  // Ensure we can cast to atomic scalar types.
  data2 = (_Atomic int)0;
  (void)(_Atomic(int *))0;

  // But that we correctly reject casts to atomic aggregate types.
  struct S { int a; } s;
  struct T { int a; };
  (void)(_Atomic struct T)s; // expected-error {{used type 'struct T' where arithmetic or pointer type is required}}
}

// Test if we can handle an _Atomic qualified integer in a switch statement.
void func_19(void) {
  _Atomic int a = 0;
  switch (a) { }
}

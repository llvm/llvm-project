// RUN: %clang_cc1 %s -verify -fsyntax-only
// RUN: %clang_cc1 %s -verify=off -fsyntax-only -Wno-atomic-access
// off-no-diagnostics

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
  x.val = 12;   // expected-error {{accessing a member of an atomic structure or union is undefined behavior}}
  xp->val = 12; // expected-error {{accessing a member of an atomic structure or union is undefined behavior}}

  _Atomic union {
    int ival;
    float fval;
  } y, *yp;
  y.ival = 12;     // expected-error {{accessing a member of an atomic structure or union is undefined behavior}}
  yp->fval = 1.2f; // expected-error {{accessing a member of an atomic structure or union is undefined behavior}}

  // RHS member access.
  int xval = x.val; // expected-error {{accessing a member of an atomic structure or union is undefined behavior}}
  xval = xp->val;   // expected-error {{accessing a member of an atomic structure or union is undefined behavior}}
  int yval = y.ival; // expected-error {{accessing a member of an atomic structure or union is undefined behavior}}
  yval = yp->ival;   // expected-error {{accessing a member of an atomic structure or union is undefined behavior}}

  // Using the type specifier instead of the type qualifier.
  _Atomic(struct { int val; }) z;
  z.val = 12;       // expected-error {{accessing a member of an atomic structure or union is undefined behavior}}
  int zval = z.val; // expected-error {{accessing a member of an atomic structure or union is undefined behavior}}

  // Don't diagnose in an unevaluated context, however.
  (void)sizeof(x.val);
  (void)sizeof(xp->val);
  (void)sizeof(y.ival);
  (void)sizeof(yp->ival);
}

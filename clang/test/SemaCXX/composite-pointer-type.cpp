// RUN: %clang_cc1 -fsyntax-only -verify %s

class Base { };
class Derived1 : public Base { };
class Derived2 : public Base { };

void f0(volatile Base *b, Derived1 *d1, const Derived2 *d2) {
  if (b > d1)
    return;
  if (d1 <= b)
    return;
  if (b > d2)
    return;
  if (d1 >= d2) // expected-error{{comparison of distinct}}
    return;
}

void f1(volatile Base *b, Derived1 *d1, const Derived2 *d2) {
  if (b == d1)
    return;
  if (d1 == b)
    return;
  if (b != d2)
    return;
  if (d1 == d2) // expected-error{{comparison of distinct}}
    return;
}

// PR4691
int ptrcmp1(void *a, int *b) {
  return a < b;
}
int ptrcmp2(long *a, int *b) {
  return a < b; // expected-error{{distinct}}
}

// PR5509 - Multi-level pointers
int f2() {
  typedef int *IntPtr;
  typedef IntPtr *IntPtrPtr;
  typedef IntPtr const *IntPtrConstPtr;
  IntPtrConstPtr i = 0;
  IntPtrPtr j = 0;
  return i != j;
}

// PR5763
typedef double Matrix4[4][4];

bool f(Matrix4 m1, const Matrix4 m2) {
  return m1 != m2;
}

// PR6346
bool f1(bool b, void **p, const void **q) {
  if (p == q)
    return false;

  return b? p : q;
}

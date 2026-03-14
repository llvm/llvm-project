// RUN: %clangxx -fsanitize=null -fno-sanitize-recover=null %s -O3 -o %t
// RUN: not %run %t l 2>&1 | FileCheck %s --check-prefix=CHECK-LOAD
// RUN: not %run %t s 2>&1 | FileCheck %s --check-prefix=CHECK-STORE
// RUN: not %run %t r 2>&1 | FileCheck %s --check-prefix=CHECK-REFERENCE
// RUN: not %run %t m 2>&1 | FileCheck %s --check-prefix=CHECK-MEMBER
// RUN: not %run %t f 2>&1 | FileCheck %s --check-prefix=CHECK-MEMFUN
// RUN: not %run %t t 2>&1 | FileCheck %s --check-prefix=CHECK-VCALL
// RUN: not %run %t u 2>&1 | FileCheck %s --check-prefix=CHECK-VCALL2

struct S {
  int f() { return 0; }
  int k;
};

struct T {
  virtual int v() { return 1; }
};

struct U : T {
  virtual int v() { return 2; }
};

int main(int, char **argv) {
  int *p = 0;
  S *s = 0;
  T *t = 0;
  U *u = 0;

  (void)*p; // ok!
  (void)*t; // ok!
  (void)*u; // ok!

  switch (argv[1][0]) {
  case 'l':
    // CHECK-LOAD: null.cpp:[[@LINE+1]]:12: runtime error: load of null pointer of type 'int'
    return *p;
  case 's':
    // CHECK-STORE: null.cpp:[[@LINE+1]]:5: runtime error: store to null pointer of type 'int'
    *p = 1;
    break;
  case 'r':
    // CHECK-REFERENCE: null.cpp:[[@LINE+1]]:15: runtime error: reference binding to null pointer of type 'int'
    {int &r = *p;}
    break;
  case 'm':
    // CHECK-MEMBER: null.cpp:[[@LINE+1]]:15: runtime error: member access within null pointer of type 'S'
    return s->k;
  case 'f':
    // CHECK-MEMFUN: null.cpp:[[@LINE+1]]:15: runtime error: member call on null pointer of type 'S'
    return s->f();
  case 't':
    // CHECK-VCALL: null.cpp:[[@LINE+1]]:15: runtime error: member call on null pointer of type 'T'
    return t->v();
  case 'u':
    // CHECK-VCALL2: null.cpp:[[@LINE+1]]:15: runtime error: member call on null pointer of type 'U'
    return u->v();
  }
}

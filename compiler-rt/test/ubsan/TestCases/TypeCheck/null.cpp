// RUN: %clangxx -fsanitize=null %s -O3 -o %t
// RUN: %run %t l 2>&1 | FileCheck %s --check-prefix=CHECK-LOAD
// RUN: %expect_crash %run %t s 2>&1 | FileCheck %s --check-prefix=CHECK-STORE
// RUN: %run %t r 2>&1 | FileCheck %s --check-prefix=CHECK-REFERENCE
// RUN: %run %t m 2>&1 | FileCheck %s --check-prefix=CHECK-MEMBER
// RUN: %run %t f 2>&1 | FileCheck %s --check-prefix=CHECK-MEMFUN

struct S {
  int f() { return 0; }
  int k;
};

int main(int, char **argv) {
  int *p = 0;
  S *s = 0;

  (void)*p; // ok!

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
  }
}

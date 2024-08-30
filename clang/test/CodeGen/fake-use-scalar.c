// RUN: %clang_cc1 %s -O2 -emit-llvm -fextend-lifetimes -o - | FileCheck %s
// Make sure we don't generate fake.use for non-scalar variables.
// Make sure we don't generate fake.use for volatile variables
// and parameters even when they are scalar.

struct A {
  unsigned long t;
  char c[1024];
  unsigned char r[32];
};


int foo(volatile int param)
{
  struct A s;
  volatile int vloc;
  struct A v[128];
  char c[33];
  return 0;
}

// CHECK-NOT:  fake.use

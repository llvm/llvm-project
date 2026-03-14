// RUN: %clang_cc1 -triple i386-unknown-unknown %s -O3 -emit-llvm -o - | FileCheck -check-prefix=CHECK-C %s
// RUN: %clang_cc1 -triple i386-unknown-unknown -x c++ %s -O3 -emit-llvm -o - | FileCheck -check-prefix=CHECK-CXX %s
// CHECK-C: ret i32 6
// CHECK-CXX: ret i32 7

// This test case illustrates a peculiarity of the promotion of
// enumeration types in C and C++. In particular, the enumeration type
// "z" below promotes to an unsigned int in C but int in C++.
static enum { foo, bar = 1U } z;

int main (void)
{
  int r = 0;

  if (bar - 2 < 0)
    r += 4;
  if (foo - 1 < 0)
    r += 2;
  if (z - 1 < 0)
    r++;

  return r;
}

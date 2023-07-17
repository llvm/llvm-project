// RUN: %clang_cc1 %s -emit-llvm -o - | FileCheck %s
// Radar 8288710: A small aggregate can be passed as an integer.  Make sure
// we don't get an error with "input constraint with a matching output
// constraint of incompatible type!" 

struct wrapper {
  int i;
};

// CHECK: xyz
int test(int i) {
  struct wrapper w;
  w.i = i;
  __asm__("xyz" : "=r" (w) : "0" (w));
  return w.i;
}

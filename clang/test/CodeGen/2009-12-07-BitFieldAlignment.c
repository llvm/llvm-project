// RUN: %clang_cc1 -triple i686-apple-darwin %s -emit-llvm -o - | FileCheck %s
// Set alignment on bitfield accesses.

struct S {
  int a, b;
  void *c;
  unsigned d : 8;
  unsigned e : 8;
};

void f0(struct S *a) {
// CHECK: load {{.*}}, align 4
// CHECK: store {{.*}}, align 4
  a->e = 0;
}

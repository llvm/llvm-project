// REQUIRES: hexagon-registered-target
// RUN: %clang_cc1 -emit-llvm -O2 -o - -triple hexagon-unknown-elf %s | FileCheck %s
// This unit test validates that the store to "dst" variable needs to be eliminated.

// CHECK: @brev_store_elimination_test1
// CHECK: llvm.hexagon.L2.loadri.pbr
// CHECK-NOT: store

int *brev_store_elimination_test1(int *ptr, int mod) {
  int dst = 100;
  return __builtin_brev_ldw(ptr, &dst, mod);
}

// CHECK: @brev_store_elimination_test2
// CHECK: llvm.hexagon.L2.loadri.pbr
// CHECK-NOT: store
extern int add(int a);
int brev_store_elimination_test2(int *ptr, int mod) {
  int dst = 100;
  __builtin_brev_ldw(ptr, &dst, mod);
  return add(dst);
}

// CHECK: @brev_store_elimination_test3
// CHECK: llvm.hexagon.L2.loadri.pbr
// CHECK-NOT: store
int brev_store_elimination_test3(int *ptr, int mod, int inc) {
  int dst = 100;
  for (int i = 0; i < inc; ++i) {
    __builtin_brev_ldw(ptr, &dst, mod);
    dst = add(dst);
  }
  return dst;
}

// brev_store_elimination_test4 validates the fact that we are not deleting the
// stores if the value is passed by reference later.
// CHECK: @brev_store_elimination_test4
// CHECK: llvm.hexagon.L2.loadri.pbr
// CHECK: store
extern int sub(int *a);
int brev_store_elimination_test4(int *ptr, int mod) {
  int dst = 100;
  __builtin_brev_ldw(ptr, &dst, mod);
  return sub(&dst);
}

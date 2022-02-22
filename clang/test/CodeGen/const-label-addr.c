// RUN: %clang_cc1 %s -emit-llvm -o - | FileCheck %s
// REQUIRES: asserts

// CHECK: @a.a = internal global i8* blockaddress(@a, %A)
int a(void) {
A:;static void* a = &&A;
}

// PR14005
// CHECK: @b.ar = internal global {{.*}} sub (i{{..}} ptrtoint (i8* blockaddress(@b, %l2) to i{{..}}), i{{..}} ptrtoint (i8* blockaddress(@b, %l1) to i{{..}}))
int b(void) {
  static int ar = &&l2 - &&l1;
l1:
  return 10;
l2:
  return 11;
}

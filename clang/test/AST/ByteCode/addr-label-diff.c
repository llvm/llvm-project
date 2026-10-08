// RUN: %clang_cc1 -triple x86_64 -emit-llvm %s -o -                                         | FileCheck %s
// RUN: %clang_cc1 -triple x86_64 -emit-llvm %s -o - -fexperimental-new-constant-interpreter | FileCheck %s

typedef __typeof((int*) 0 - (int*) 0) intptr_t;

// CHECK: @f1.l0 = internal global i64 ptrtoint (ptr @f1 to i64)
void f1(void) { static intptr_t l0 = (intptr_t) f1; }

// CHECK: @FoldableAddrLabelDiff.x = internal global i64 sub (i64 ptrtoint (ptr blockaddress(@FoldableAddrLabelDiff, %a) to i64), i64 ptrtoint (ptr blockaddress(@FoldableAddrLabelDiff, %b) to i64)), align 8
void FoldableAddrLabelDiff() { static long x = (long)&&a-(long)&&b; a:b:return;}

// CHECK: @c.ar = internal global {{.*}} sub (i{{..}} ptrtoint (ptr blockaddress(@c, %l2) to i{{..}}), i{{..}} ptrtoint (ptr blockaddress(@c, %l1) to i{{..}}))
int c(void) {
  static int ar = &&l2 - &&l1;
l1:
  return 10;
l2:
  return 11;
}

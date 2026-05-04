// RUN: %clang_cc1 %s -emit-llvm -debug-info-kind=limited -o - | FileCheck %s
// CHECK: DW_TAG_pointer_type
// CHECK-NOT: {"char"}

char i = 1;
void foo(void) {
  char *cp = &i;
}


// RUN: %clang_cc1 -O2 -triple s390x-linux -emit-llvm -o - %s | FileCheck %s

int foo_012(int x) {
// CHECK-LABEL: @foo_012
// CHECK: = tail call { i32, i32 } asm "ahi $0,42\0A", "=d,={@cc},0"(i32 %x)
  int cc;
  asm ("ahi %[x],42\n" : [x] "+d"(x), "=@cc" (cc));
  return cc == 0 || cc == 1 || cc == 2 ? 42 : 0;
}

int foo_013(int x) {
// CHECK-LABEL: @foo_013
// CHECK: = tail call { i32, i32 } asm "ahi $0,42\0A", "=d,={@cc},0"(i32 %x)
  int cc;
  asm ("ahi %[x],42\n" : [x] "+d"(x), "=@cc" (cc));
  return cc == 0 || cc == 1 || cc == 3 ? 42 : 0;
}

int foo_023(int x) {
// CHECK-LABEL: @foo_023
// CHECK: = tail call { i32, i32 } asm "ahi $0,42\0A", "=d,={@cc},0"(i32 %x)
  int cc;
  asm ("ahi %[x],42\n" : [x] "+d"(x), "=@cc" (cc));
  return cc == 0 || cc == 2 || cc == 3 ? 42 : 0;
}

int foo_123(int x) {
// CHECK-LABEL: @foo_123
// CHECK: = tail call { i32, i32 } asm "ahi $0,42\0A", "=d,={@cc},0"(i32 %x)
  int cc;
  asm ("ahi %[x],42\n" : [x] "+d"(x), "=@cc" (cc));
  return cc == 1 || cc == 2 || cc == 3 ? 42 : 0;
}

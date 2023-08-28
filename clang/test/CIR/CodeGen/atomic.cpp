// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

typedef struct _a {
  _Atomic(int) d;
} at;

void m() { at y; }

// CHECK: !ty_22_a22 = !cir.struct<struct "_a" {!s32i}>
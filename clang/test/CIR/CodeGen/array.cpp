// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

void a0() {
  int a[10];
}

// CHECK: func @a0() {
// CHECK-NEXT:   %0 = cir.alloca !cir.array<i32 x 10>, cir.ptr <!cir.array<i32 x 10>>, ["a", uninitialized] {alignment = 16 : i64}
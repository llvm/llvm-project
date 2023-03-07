// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

void foo() {
  int a = 0;
  a = 1;
}

//      CHECK: cir.func @foo() {
// CHECK-NEXT:   %0 = cir.alloca i32, cir.ptr <i32>, ["a", init] {alignment = 4 : i64}
// CHECK-NEXT:   %1 = cir.const(0 : i32) : i32
// CHECK-NEXT:   cir.store %1, %0 : i32, cir.ptr <i32>
// CHECK-NEXT:   %2 = cir.const(1 : i32) : i32
// CHECK-NEXT:   cir.store %2, %0 : i32, cir.ptr <i32>
// CHECK-NEXT:   cir.return
// CHECK-NEXT: }


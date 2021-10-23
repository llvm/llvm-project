// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

int foo(int i) {
  return i;
}

// CHECK: module  {
// CHECK-NEXT: func @foo(%arg0: i32) -> i32 {
// CHECK-NEXT:   %0 = cir.alloca i32, cir.ptr <i32>, [paraminit]
// CHECK-NEXT:   cir.store %arg0, %0 : i32, cir.ptr <i32>
// CHECK-NEXT:   %1 = cir.load %0 lvalue_to_rvalue : cir.ptr <i32>, i32
// CHECK-NEXT:   cir.return %1 : i32
// CHECK-NEXT: }

int f2() { return 3; }

// CHECK: func @f2() -> i32 {
// CHECK-NEXT: %0 = cir.cst(3 : i32) : i32
// CHECK-NEXT: cir.return %0 : i32

int f3() {
  int i = 3;
  return i;
}

// CHECK: func @f3() -> i32 {
// CHECK-NEXT:   %0 = cir.alloca i32, cir.ptr <i32>, [cinit]
// CHECK-NEXT:   %1 = cir.cst(3 : i32) : i32
// CHECK-NEXT:   cir.store %1, %0 : i32, cir.ptr <i32>

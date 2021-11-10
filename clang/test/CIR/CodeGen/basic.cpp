// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

int *p0() {
  int *p = nullptr;
  return p;
}

// CHECK: func @p0() -> !cir.ptr<i32> {
// CHECK: %1 = cir.cst(#cir.null : !cir.ptr<i32>) : !cir.ptr<i32>
// CHECK: cir.store %1, %0 : !cir.ptr<i32>, cir.ptr <!cir.ptr<i32>>

int *p1() {
  int *p;
  p = nullptr;
  return p;
}

// CHECK: func @p1() -> !cir.ptr<i32> {
// CHECK: %0 = cir.alloca !cir.ptr<i32>, cir.ptr <!cir.ptr<i32>>, [uninitialized]
// CHECK: %1 = cir.cst(#cir.null : !cir.ptr<i32>) : !cir.ptr<i32>
// CHECK: cir.store %1, %0 : !cir.ptr<i32>, cir.ptr <!cir.ptr<i32>>

int *p2() {
  int *p = nullptr;
  int x = 0;
  p = &x;
  return p;
}

// CHECK: func @p2() -> !cir.ptr<i32> {
// CHECK:     %0 = cir.alloca i32, cir.ptr <i32>, [cinit]
// CHECK:     %1 = cir.alloca !cir.ptr<i32>, cir.ptr <!cir.ptr<i32>>, [cinit]
// CHECK:     cir.store %0, %1 : !cir.ptr<i32>, cir.ptr <!cir.ptr<i32>>

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s
// XFAIL: *

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
  *p = 42;
  return p;
}

// CHECK: func @p2() -> !cir.ptr<i32> {
// CHECK:     %0 = cir.alloca i32, cir.ptr <i32>, [cinit]
// CHECK:     %1 = cir.alloca !cir.ptr<i32>, cir.ptr <!cir.ptr<i32>>, [cinit]
// CHECK:     cir.store %0, %1 : !cir.ptr<i32>, cir.ptr <!cir.ptr<i32>>
// CHECK:     %4 = cir.cst(42 : i32) : i32
// CHECK-NEXT:     %5 = cir.load %1 lvalue_to_rvalue : cir.ptr <!cir.ptr<i32>>, !cir.ptr<i32>
// CHECK-NEXT:     cir.store %4, %5 : i32, cir.ptr <i32>

void b0() { bool x = true, y = false; }

// CHECK: func @b0() {
// CHECK: %2 = cir.cst(true) : !cir.bool
// CHECK: %3 = cir.cst(false) : !cir.bool

void b1(int a) { bool b = a; }

// CHECK: func @b1(%arg0: i32) {
// CHECK: %2 = cir.load %1 lvalue_to_rvalue : cir.ptr <i32>, i32
// CHECK: %3 = cir.cast(int_to_bool, %2 : i32), !cir.bool
// CHECK: cir.store %3, %0 : !cir.bool, cir.ptr <!cir.bool>

int if0(int a) {
  int x = 0;
  if (a) {
    x = 3;
  } else {
    x = 4;
  }
  return x;
}

// CHECK: func @if0(%arg0: i32) -> i32 {
// CHECK: %4 = cir.cast(int_to_bool, %3 : i32), !cir.bool
// CHECK-NEXT: cir.if %4 {
// CHECK-NEXT:   %6 = cir.cst(3 : i32) : i32
// CHECK-NEXT:   cir.store %6, %0 : i32, cir.ptr <i32>
// CHECK-NEXT: } else {
// CHECK-NEXT:   %6 = cir.cst(4 : i32) : i32
// CHECK-NEXT:   cir.store %6, %0 : i32, cir.ptr <i32>
// CHECK-NEXT: }

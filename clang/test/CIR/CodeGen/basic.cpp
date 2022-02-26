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
// CHECK: %0 = cir.alloca !cir.ptr<i32>, cir.ptr <!cir.ptr<i32>>, ["p", uninitialized]
// CHECK: %1 = cir.cst(#cir.null : !cir.ptr<i32>) : !cir.ptr<i32>
// CHECK: cir.store %1, %0 : !cir.ptr<i32>, cir.ptr <!cir.ptr<i32>>

int *p2() {
  int *p = nullptr;
  {
    int x = 0;
    p = &x;
    *p = 42;
  }
  *p = 42;
  return p;
}

// CHECK: func @p2() -> !cir.ptr<i32> {
// CHECK-NEXT:    %0 = cir.alloca !cir.ptr<i32>, cir.ptr <!cir.ptr<i32>>, ["p", cinit] {alignment = 8 : i64}
// CHECK-NEXT:    %1 = cir.cst(#cir.null : !cir.ptr<i32>) : !cir.ptr<i32>
// CHECK-NEXT:    cir.store %1, %0 : !cir.ptr<i32>, cir.ptr <!cir.ptr<i32>>
// CHECK-NEXT:    cir.scope {
// CHECK-NEXT:      %5 = cir.alloca i32, cir.ptr <i32>, ["x", cinit] {alignment = 4 : i64}
// CHECK-NEXT:      %6 = cir.cst(0 : i32) : i32
// CHECK-NEXT:      cir.store %6, %5 : i32, cir.ptr <i32>
// CHECK-NEXT:      cir.store %5, %0 : !cir.ptr<i32>, cir.ptr <!cir.ptr<i32>>
// CHECK-NEXT:      %7 = cir.cst(42 : i32) : i32
// CHECK-NEXT:      %8 = cir.load deref %0 : cir.ptr <!cir.ptr<i32>>, !cir.ptr<i32>
// CHECK-NEXT:      cir.store %7, %8 : i32, cir.ptr <i32>
// CHECK-NEXT:    }
// CHECK-NEXT:    %2 = cir.cst(42 : i32) : i32
// CHECK-NEXT:    %3 = cir.load deref %0 : cir.ptr <!cir.ptr<i32>>, !cir.ptr<i32>
// CHECK-NEXT:    cir.store %2, %3 : i32, cir.ptr <i32>
// CHECK-NEXT:    %4 = cir.load %0 : cir.ptr <!cir.ptr<i32>>, !cir.ptr<i32>
// CHECK-NEXT:    cir.return %4 : !cir.ptr<i32>
// CHECK-NEXT:  }

void b0() { bool x = true, y = false; }

// CHECK: func @b0() {
// CHECK: %2 = cir.cst(true) : !cir.bool
// CHECK: %3 = cir.cst(false) : !cir.bool

void b1(int a) { bool b = a; }

// CHECK: func @b1(%arg0: i32 loc({{.*}})) {
// CHECK: %2 = cir.load %0 : cir.ptr <i32>, i32
// CHECK: %3 = cir.cast(int_to_bool, %2 : i32), !cir.bool
// CHECK: cir.store %3, %1 : !cir.bool, cir.ptr <!cir.bool>

int if0(int a) {
  int x = 0;
  if (a) {
    x = 3;
  } else {
    x = 4;
  }
  return x;
}

// CHECK: func @if0(%arg0: i32 loc({{.*}})) -> i32 {
// CHECK: cir.scope {
// CHECK:   %4 = cir.load %0 : cir.ptr <i32>, i32
// CHECK:   %5 = cir.cast(int_to_bool, %4 : i32), !cir.bool
// CHECK-NEXT:   cir.if %5 {
// CHECK-NEXT:     %6 = cir.cst(3 : i32) : i32
// CHECK-NEXT:     cir.store %6, %1 : i32, cir.ptr <i32>
// CHECK-NEXT:   } else {
// CHECK-NEXT:     %6 = cir.cst(4 : i32) : i32
// CHECK-NEXT:     cir.store %6, %1 : i32, cir.ptr <i32>
// CHECK-NEXT:   }
// CHECK: }

int if1(int a, bool b, bool c) {
  int x = 0;
  if (a) {
    x = 3;
    if (b) {
      x = 8;
    }
  } else {
    if (c) {
      x = 14;
    }
    x = 4;
  }
  return x;
}

// CHECK: func @if1(%arg0: i32 loc({{.*}}), %arg1: !cir.bool loc({{.*}}), %arg2: !cir.bool loc({{.*}})) -> i32 {
// CHECK: cir.scope {
// CHECK:   %6 = cir.load %0 : cir.ptr <i32>, i32
// CHECK:   %7 = cir.cast(int_to_bool, %6 : i32), !cir.bool
// CHECK:   cir.if %7 {
// CHECK:     %8 = cir.cst(3 : i32) : i32
// CHECK:     cir.store %8, %3 : i32, cir.ptr <i32>
// CHECK:     cir.scope {
// CHECK:       %9 = cir.load %1 : cir.ptr <!cir.bool>, !cir.bool
// CHECK-NEXT:       cir.if %9 {
// CHECK-NEXT:         %10 = cir.cst(8 : i32) : i32
// CHECK-NEXT:         cir.store %10, %3 : i32, cir.ptr <i32>
// CHECK-NEXT:       }
// CHECK:     }
// CHECK:   } else {
// CHECK:     cir.scope {
// CHECK:       %9 = cir.load %2 : cir.ptr <!cir.bool>, !cir.bool
// CHECK-NEXT:       cir.if %9 {
// CHECK-NEXT:         %10 = cir.cst(14 : i32) : i32
// CHECK-NEXT:         cir.store %10, %3 : i32, cir.ptr <i32>
// CHECK-NEXT:       }
// CHECK:     }
// CHECK:     %8 = cir.cst(4 : i32) : i32
// CHECK:     cir.store %8, %3 : i32, cir.ptr <i32>
// CHECK:   }
// CHECK: }

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s
// XFAIL: *

int *p0() {
  int *p = nullptr;
  return p;
}

// CHECK: cir.func @_Z2p0v() -> !cir.ptr<i32> {
// CHECK: %1 = cir.alloca !cir.ptr<i32>, cir.ptr <!cir.ptr<i32>>, ["p", cinit]
// CHECK: %2 = cir.cst(#cir.null : !cir.ptr<i32>) : !cir.ptr<i32>
// CHECK: cir.store %2, %1 : !cir.ptr<i32>, cir.ptr <!cir.ptr<i32>>

int *p1() {
  int *p;
  p = nullptr;
  return p;
}

// CHECK: cir.func @_Z2p1v() -> !cir.ptr<i32> {
// CHECK: %1 = cir.alloca !cir.ptr<i32>, cir.ptr <!cir.ptr<i32>>, ["p", uninitialized]
// CHECK: %2 = cir.cst(#cir.null : !cir.ptr<i32>) : !cir.ptr<i32>
// CHECK: cir.store %2, %1 : !cir.ptr<i32>, cir.ptr <!cir.ptr<i32>>

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

// CHECK: cir.func @_Z2p2v() -> !cir.ptr<i32> {
// CHECK-NEXT:  %0 = cir.alloca !cir.ptr<i32>, cir.ptr <!cir.ptr<i32>>, ["__retval", uninitialized] {alignment = 8 : i64}
// CHECK-NEXT:  %1 = cir.alloca !cir.ptr<i32>, cir.ptr <!cir.ptr<i32>>, ["p", cinit] {alignment = 8 : i64}
// CHECK-NEXT:  %2 = cir.cst(#cir.null : !cir.ptr<i32>) : !cir.ptr<i32>
// CHECK-NEXT:  cir.store %2, %1 : !cir.ptr<i32>, cir.ptr <!cir.ptr<i32>>
// CHECK-NEXT:  cir.scope {
// CHECK-NEXT:    %7 = cir.alloca i32, cir.ptr <i32>, ["x", cinit] {alignment = 4 : i64}
// CHECK-NEXT:    %8 = cir.cst(0 : i32) : i32
// CHECK-NEXT:    cir.store %8, %7 : i32, cir.ptr <i32>
// CHECK-NEXT:    cir.store %7, %1 : !cir.ptr<i32>, cir.ptr <!cir.ptr<i32>>
// CHECK-NEXT:    %9 = cir.cst(42 : i32) : i32
// CHECK-NEXT:    %10 = cir.load deref %1 : cir.ptr <!cir.ptr<i32>>, !cir.ptr<i32>
// CHECK-NEXT:    cir.store %9, %10 : i32, cir.ptr <i32>
// CHECK-NEXT:  } loc(#[[locScope:loc[0-9]+]])
// CHECK-NEXT:  %3 = cir.cst(42 : i32) : i32
// CHECK-NEXT:  %4 = cir.load deref %1 : cir.ptr <!cir.ptr<i32>>, !cir.ptr<i32>
// CHECK-NEXT:  cir.store %3, %4 : i32, cir.ptr <i32>
// CHECK-NEXT:  %5 = cir.load %1 : cir.ptr <!cir.ptr<i32>>, !cir.ptr<i32>
// CHECK-NEXT:  cir.store %5, %0 : !cir.ptr<i32>, cir.ptr <!cir.ptr<i32>>
// CHECK-NEXT:  %6 = cir.load %0 : cir.ptr <!cir.ptr<i32>>, !cir.ptr<i32>
// CHECK-NEXT:  cir.return %6 : !cir.ptr<i32>

void b0() { bool x = true, y = false; }

// CHECK: cir.func @_Z2b0v() {
// CHECK: %2 = cir.cst(true) : !cir.bool
// CHECK: %3 = cir.cst(false) : !cir.bool

void b1(int a) { bool b = a; }

// CHECK: cir.func @_Z2b1i(%arg0: i32 loc({{.*}})) {
// CHECK: %2 = cir.load %0 : cir.ptr <i32>, i32
// CHECK: %3 = cir.cast(int_to_bool, %2 : i32), !cir.bool
// CHECK: cir.store %3, %1 : !cir.bool, cir.ptr <!cir.bool>

void if0(int a) {
  int x = 0;
  if (a) {
    x = 3;
  } else {
    x = 4;
  }
}

// CHECK: cir.func @_Z3if0i(%arg0: i32 loc({{.*}}))
// CHECK: cir.scope {
// CHECK:   %3 = cir.load %0 : cir.ptr <i32>, i32
// CHECK:   %4 = cir.cast(int_to_bool, %3 : i32), !cir.bool
// CHECK-NEXT:   cir.if %4 {
// CHECK-NEXT:     %5 = cir.cst(3 : i32) : i32
// CHECK-NEXT:     cir.store %5, %1 : i32, cir.ptr <i32>
// CHECK-NEXT:   } else {
// CHECK-NEXT:     %5 = cir.cst(4 : i32) : i32
// CHECK-NEXT:     cir.store %5, %1 : i32, cir.ptr <i32>
// CHECK-NEXT:   }
// CHECK: }

void if1(int a, bool b, bool c) {
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
}

// CHECK: cir.func @_Z3if1ibb(%arg0: i32 loc({{.*}}), %arg1: !cir.bool loc({{.*}}), %arg2: !cir.bool loc({{.*}}))
// CHECK: cir.scope {
// CHECK:   %5 = cir.load %0 : cir.ptr <i32>, i32
// CHECK:   %6 = cir.cast(int_to_bool, %5 : i32), !cir.bool
// CHECK:   cir.if %6 {
// CHECK:     %7 = cir.cst(3 : i32) : i32
// CHECK:     cir.store %7, %3 : i32, cir.ptr <i32>
// CHECK:     cir.scope {
// CHECK:       %8 = cir.load %1 : cir.ptr <!cir.bool>, !cir.bool
// CHECK-NEXT:       cir.if %8 {
// CHECK-NEXT:         %9 = cir.cst(8 : i32) : i32
// CHECK-NEXT:         cir.store %9, %3 : i32, cir.ptr <i32>
// CHECK-NEXT:       }
// CHECK:     }
// CHECK:   } else {
// CHECK:     cir.scope {
// CHECK:       %8 = cir.load %2 : cir.ptr <!cir.bool>, !cir.bool
// CHECK-NEXT:       cir.if %8 {
// CHECK-NEXT:         %9 = cir.cst(14 : i32) : i32
// CHECK-NEXT:         cir.store %9, %3 : i32, cir.ptr <i32>
// CHECK-NEXT:       }
// CHECK:     }
// CHECK:     %7 = cir.cst(4 : i32) : i32
// CHECK:     cir.store %7, %3 : i32, cir.ptr <i32>
// CHECK:   }
// CHECK: }

// CHECK: #[[locScope]] = loc(fused["{{.*}}basic.cpp":27:3, "{{.*}}basic.cpp":31:3])

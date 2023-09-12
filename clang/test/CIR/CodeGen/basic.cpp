// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

int *p0() {
  int *p = nullptr;
  return p;
}

// CHECK: cir.func @_Z2p0v() -> !cir.ptr<!s32i>
// CHECK: %1 = cir.alloca !cir.ptr<!s32i>, cir.ptr <!cir.ptr<!s32i>>, ["p", init]
// CHECK: %2 = cir.const(#cir.ptr<null> : !cir.ptr<!s32i>) : !cir.ptr<!s32i>
// CHECK: cir.store %2, %1 : !cir.ptr<!s32i>, cir.ptr <!cir.ptr<!s32i>>

int *p1() {
  int *p;
  p = nullptr;
  return p;
}

// CHECK: cir.func @_Z2p1v() -> !cir.ptr<!s32i>
// CHECK: %1 = cir.alloca !cir.ptr<!s32i>, cir.ptr <!cir.ptr<!s32i>>, ["p"]
// CHECK: %2 = cir.const(#cir.ptr<null> : !cir.ptr<!s32i>) : !cir.ptr<!s32i>
// CHECK: cir.store %2, %1 : !cir.ptr<!s32i>, cir.ptr <!cir.ptr<!s32i>>

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

// CHECK: cir.func @_Z2p2v() -> !cir.ptr<!s32i>
// CHECK-NEXT:  %0 = cir.alloca !cir.ptr<!s32i>, cir.ptr <!cir.ptr<!s32i>>, ["__retval"] {alignment = 8 : i64}
// CHECK-NEXT:  %1 = cir.alloca !cir.ptr<!s32i>, cir.ptr <!cir.ptr<!s32i>>, ["p", init] {alignment = 8 : i64}
// CHECK-NEXT:  %2 = cir.const(#cir.ptr<null> : !cir.ptr<!s32i>) : !cir.ptr<!s32i>
// CHECK-NEXT:  cir.store %2, %1 : !cir.ptr<!s32i>, cir.ptr <!cir.ptr<!s32i>>
// CHECK-NEXT:  cir.scope {
// CHECK-NEXT:    %7 = cir.alloca !s32i, cir.ptr <!s32i>, ["x", init] {alignment = 4 : i64}
// CHECK-NEXT:    %8 = cir.const(#cir.int<0> : !s32i) : !s32i
// CHECK-NEXT:    cir.store %8, %7 : !s32i, cir.ptr <!s32i>
// CHECK-NEXT:    cir.store %7, %1 : !cir.ptr<!s32i>, cir.ptr <!cir.ptr<!s32i>>
// CHECK-NEXT:    %9 = cir.const(#cir.int<42> : !s32i) : !s32i
// CHECK-NEXT:    %10 = cir.load deref %1 : cir.ptr <!cir.ptr<!s32i>>, !cir.ptr<!s32i>
// CHECK-NEXT:    cir.store %9, %10 : !s32i, cir.ptr <!s32i>
// CHECK-NEXT:  } loc(#[[locScope:loc[0-9]+]])
// CHECK-NEXT:  %3 = cir.const(#cir.int<42> : !s32i) : !s32i
// CHECK-NEXT:  %4 = cir.load deref %1 : cir.ptr <!cir.ptr<!s32i>>, !cir.ptr<!s32i>
// CHECK-NEXT:  cir.store %3, %4 : !s32i, cir.ptr <!s32i>
// CHECK-NEXT:  %5 = cir.load %1 : cir.ptr <!cir.ptr<!s32i>>, !cir.ptr<!s32i>
// CHECK-NEXT:  cir.store %5, %0 : !cir.ptr<!s32i>, cir.ptr <!cir.ptr<!s32i>>
// CHECK-NEXT:  %6 = cir.load %0 : cir.ptr <!cir.ptr<!s32i>>, !cir.ptr<!s32i>
// CHECK-NEXT:  cir.return %6 : !cir.ptr<!s32i>

void b0() { bool x = true, y = false; }

// CHECK: cir.func @_Z2b0v()
// CHECK: %2 = cir.const(#true) : !cir.bool
// CHECK: %3 = cir.const(#false) : !cir.bool

void b1(int a) { bool b = a; }

// CHECK: cir.func @_Z2b1i(%arg0: !s32i loc({{.*}}))
// CHECK: %2 = cir.load %0 : cir.ptr <!s32i>, !s32i
// CHECK: %3 = cir.cast(int_to_bool, %2 : !s32i), !cir.bool
// CHECK: cir.store %3, %1 : !cir.bool, cir.ptr <!cir.bool>

void if0(int a) {
  int x = 0;
  if (a) {
    x = 3;
  } else {
    x = 4;
  }
}

// CHECK: cir.func @_Z3if0i(%arg0: !s32i loc({{.*}}))
// CHECK: cir.scope {
// CHECK:   %3 = cir.load %0 : cir.ptr <!s32i>, !s32i
// CHECK:   %4 = cir.cast(int_to_bool, %3 : !s32i), !cir.bool
// CHECK-NEXT:   cir.if %4 {
// CHECK-NEXT:     %5 = cir.const(#cir.int<3> : !s32i) : !s32i
// CHECK-NEXT:     cir.store %5, %1 : !s32i, cir.ptr <!s32i>
// CHECK-NEXT:   } else {
// CHECK-NEXT:     %5 = cir.const(#cir.int<4> : !s32i) : !s32i
// CHECK-NEXT:     cir.store %5, %1 : !s32i, cir.ptr <!s32i>
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

// CHECK: cir.func @_Z3if1ibb(%arg0: !s32i loc({{.*}}), %arg1: !cir.bool loc({{.*}}), %arg2: !cir.bool loc({{.*}}))
// CHECK: cir.scope {
// CHECK:   %5 = cir.load %0 : cir.ptr <!s32i>, !s32i
// CHECK:   %6 = cir.cast(int_to_bool, %5 : !s32i), !cir.bool
// CHECK:   cir.if %6 {
// CHECK:     %7 = cir.const(#cir.int<3> : !s32i) : !s32i
// CHECK:     cir.store %7, %3 : !s32i, cir.ptr <!s32i>
// CHECK:     cir.scope {
// CHECK:       %8 = cir.load %1 : cir.ptr <!cir.bool>, !cir.bool
// CHECK-NEXT:       cir.if %8 {
// CHECK-NEXT:         %9 = cir.const(#cir.int<8> : !s32i) : !s32i
// CHECK-NEXT:         cir.store %9, %3 : !s32i, cir.ptr <!s32i>
// CHECK-NEXT:       }
// CHECK:     }
// CHECK:   } else {
// CHECK:     cir.scope {
// CHECK:       %8 = cir.load %2 : cir.ptr <!cir.bool>, !cir.bool
// CHECK-NEXT:       cir.if %8 {
// CHECK-NEXT:         %9 = cir.const(#cir.int<14> : !s32i) : !s32i
// CHECK-NEXT:         cir.store %9, %3 : !s32i, cir.ptr <!s32i>
// CHECK-NEXT:       }
// CHECK:     }
// CHECK:     %7 = cir.const(#cir.int<4> : !s32i) : !s32i
// CHECK:     cir.store %7, %3 : !s32i, cir.ptr <!s32i>
// CHECK:   }
// CHECK: }

enum {
  um = 0,
  dois = 1,
}; // Do not crash!

extern "C" {
struct regs {
  unsigned long sp;
  unsigned long pc;
};

// Check it's not mangled.
// CHECK: cir.func @use_regs()

void use_regs() { regs r; }
}

void x() {
  const bool b0 = true;
  const bool b1 = false;
}

// CHECK: cir.func @_Z1xv()
// CHECK:   %0 = cir.alloca !cir.bool, cir.ptr <!cir.bool>, ["b0", init] {alignment = 1 : i64}
// CHECK:   %1 = cir.alloca !cir.bool, cir.ptr <!cir.bool>, ["b1", init] {alignment = 1 : i64}
// CHECK:   %2 = cir.const(#true) : !cir.bool
// CHECK:   cir.store %2, %0 : !cir.bool, cir.ptr <!cir.bool>
// CHECK:   %3 = cir.const(#false) : !cir.bool
// CHECK:   cir.store %3, %1 : !cir.bool, cir.ptr <!cir.bool>

typedef unsigned long size_type;
typedef unsigned long _Tp;

size_type max_size() {
  return size_type(~0) / sizeof(_Tp);
}

// CHECK: cir.func @_Z8max_sizev()
// CHECK:   %0 = cir.alloca !u64i, cir.ptr <!u64i>, ["__retval"] {alignment = 8 : i64}
// CHECK:   %1 = cir.const(#cir.int<0> : !s32i) : !s32i
// CHECK:   %2 = cir.unary(not, %1) : !s32i, !s32i
// CHECK:   %3 = cir.cast(integral, %2 : !s32i), !u64i
// CHECK:   %4 = cir.const(#cir.int<8> : !u64i) : !u64i
// CHECK:   %5 = cir.binop(div, %3, %4) : !u64i

// CHECK-DAG: #[[locScope]] = loc(fused[#[[locScopeA:loc[0-9]+]], #[[locScopeB:loc[0-9]+]]])
// CHECK-DAG: #[[locScopeA]] = loc("{{.*}}basic.cpp":27:3)
// CHECK-DAG: #[[locScopeB]] = loc("{{.*}}basic.cpp":31:3)

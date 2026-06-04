// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

struct EmptyLo {};
struct Val { long v; };

struct Ret {
  EmptyLo lo;
  Val hi;
};

Ret make(long x) {
  Ret r;
  r.hi.v = x;
  return r;
}

long take(Ret r) { return r.hi.v; }

long caller() {
  Ret tmp = make(99);
  return take(tmp);
}

// Negative case: guard must not fire for (SSE, INTEGER) struct.
struct TwoReg { double lo; long hi; };
TwoReg makeTwoReg(long x);
long callTwoReg() {
  TwoReg r = makeTwoReg(42);
  return r.hi;
}

// CIR: cir.func {{.*}} @_Z4makel(%{{.*}}: !s64i
// CIR-SAME: -> !s64i
// CIR: cir.const #cir.int<8> : !u64i
// CIR: cir.ptr_stride
// CIR: cir.load {{.*}} : !cir.ptr<!s64i>, !s64i
// CIR: cir.return {{.*}} : !s64i

// CIR: cir.func {{.*}} @_Z6callerv()
// CIR: cir.call @_Z4makel
// CIR-SAME: -> !s64i
// CIR: cir.const #cir.int<8> : !u64i
// CIR: cir.ptr_stride
// CIR: cir.store {{.*}} : !s64i

// LLVM: define {{.*}} @_Z4makel(i64
// LLVM: define {{.*}} @_Z6callerv()
// LLVM: call i64 @_Z4makel(i64
// LLVM: getelementptr{{.*}}i8, ptr %{{.*}}, i64 8

// CIR: cir.func{{.*}} @_Z10makeTwoRegl{{.*}} -> !rec_TwoReg

// OGCG: call { double, i64 } @_Z10makeTwoRegl(

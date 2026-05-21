// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

struct Pad8 {
  char c[8];
};

struct Val {
  long v;
};

struct Ret {
  Pad8 pad;
  Val val;
};

Ret make(long x) {
  Ret r{{0}, {x}};
  return r;
}

long take(Ret r) { return r.val.v; }

long caller() {
  Ret tmp = make(99);
  return take(tmp);
}

// Coerced 16-byte struct return: only the high eightbyte (field at +8) is
// returned in a register; CIR stores the call result at offset 8.
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
// LLVM: getelementptr i8, ptr %{{.*}}, i64 8

// OGCG: define {{.*}} @_Z6callerv()
// OGCG: call { i64, i64 } @_Z4makel(i64
// OGCG: call noundef i64 @_Z4take3Ret(i64

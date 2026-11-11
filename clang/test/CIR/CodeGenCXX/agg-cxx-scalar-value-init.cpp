// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -emit-llvm -disable-llvm-passes %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

struct S {
  int a;
};

// CIR-LABEL: cir.func {{.*}} @_Z4testb
// LLVM-LABEL: define {{.*}} @_Z4testb
// OGCG-LABEL: define {{.*}} @_Z4testb
void test(bool b) {
  // CIR: cir.if %{{.+}} {
  // CIR:   %[[MEM:.*]] = cir.get_member %{{.*}}[0] {name = "a"} : !cir.ptr<!rec_S> -> !cir.ptr<!s32i>
  // CIR:   %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
  // CIR:   cir.store align(4) %[[ZERO]], %[[MEM]] : !s32i, !cir.ptr<!s32i>
  // CIR: } else {
  // CIR:   %[[MEM2:.*]] = cir.get_member %{{.*}}[0] {name = "a"} : !cir.ptr<!rec_S> -> !cir.ptr<!s32i>
  // CIR:   %[[ZERO2:.*]] = cir.const #cir.int<0> : !s32i
  // CIR:   cir.store align(4) %[[ZERO2]], %[[MEM2]] : !s32i, !cir.ptr<!s32i>
  // CIR: }

  // LLVM: store i32 0, ptr %{{.*}}, align 4
  // LLVM: store i32 0, ptr %{{.*}}, align 4

  // OGCG: store i32 0, ptr %{{.*}}, align 4
  S s = b ? S{int()} : S{int()};
}

struct Pair {
  int a;
  int b;
};
using AS = _Atomic(Pair);

// CIR-LABEL: cir.func {{.*}} @_Z11test_atomicv
// LLVM-LABEL: define {{.*}} @_Z11test_atomicv
// OGCG-LABEL: define {{.*}} @_Z11test_atomicv
void test_atomic() {
  // CIR: %[[S:.*]] = cir.alloca "s" align(8) init : !cir.ptr<!rec_Pair>
  // CIR: %[[S_U8:.*]] = cir.cast bitcast %[[S]] : !cir.ptr<!rec_Pair> -> !cir.ptr<!u8i>
  // CIR: %[[ZERO:.*]] = cir.const #cir.int<0> : !u8i
  // CIR: %[[SIZE:.*]] = cir.const #cir.int<8> : !u64i
  // CIR: %[[S_VOID:.*]] = cir.cast bitcast %[[S_U8]] : !cir.ptr<!u8i> -> !cir.ptr<!void>
  // CIR: cir.libc.memset %[[SIZE]] bytes at %[[S_VOID]] align(8) to %[[ZERO]] : !cir.ptr<!void>, !u8i, !u64i

  // LLVM: call void @llvm.memset.p0.i64(ptr align 8 %{{.*}}, i8 0, i64 8, i1 false)

  // OGCG: call void @llvm.memset.p0.i64(ptr align 8 %{{.*}}, i8 0, i64 8, i1 false)
  AS s = AS();
}

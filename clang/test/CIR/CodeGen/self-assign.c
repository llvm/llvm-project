// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM

// This is UB, but Clang allows it and it is used in one of the
// llvm-test-suite MultiSource tests.

struct S {
  int a;
  int b;
  int c;
};

void test_self_initialize() {
  struct S s = s;
}

// CIR: cir.func{{.*}} @test_self_initialize()
//   %[[S:.*]] = cir.alloca !rec_S, !cir.ptr<!rec_S>, ["s", init]
//   cir.copy %[[S]] to %[[S]] : !cir.ptr<!rec_S>

// LLVM: define{{.*}} void @test_self_initialize()
// LLVM:   %[[S:.*]] = alloca %struct.S
// LLVM:   call void @llvm.memcpy.p0.p0.i64(ptr{{.*}} %[[S]], ptr{{.*}} %[[S]], i64 12, i1 false)

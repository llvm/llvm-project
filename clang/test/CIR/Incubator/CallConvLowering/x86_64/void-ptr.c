// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -fclangir-call-conv-lowering \
// RUN:   -emit-cir-flat -mmlir --mlir-print-ir-after=cir-call-conv-lowering %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -fclangir-call-conv-lowering \
// RUN:   -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

// CIR: cir.func {{.*}} @test_void_ptr(%arg0: !cir.ptr<!void> {{.*}}) -> !cir.ptr<!void>
// LLVM: define {{.*}}ptr @test_void_ptr(ptr %{{.*}})
// OGCG: define {{.*}}ptr @test_void_ptr(ptr {{.*}}%{{.*}})
void *test_void_ptr(void *p) {
  return p;
}

// CIR: cir.func {{.*}} @test_void_ptr_offset(%arg0: !cir.ptr<!void> {{.*}}, %arg1: !s64i {{.*}}) -> !cir.ptr<!void>
// LLVM: define {{.*}}ptr @test_void_ptr_offset(ptr %{{.*}}, i64 %{{.*}})
// OGCG: define {{.*}}ptr @test_void_ptr_offset(ptr {{.*}}%{{.*}}, i64 {{.*}}%{{.*}})
void *test_void_ptr_offset(void *p, long offset) {
  return (char*)p + offset;
}

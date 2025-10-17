// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

double fabs(double x) {
  return __builtin_fabs(x);
}

// CIR: {{.*}} = cir.fabs {{.*}} : !cir.double
// LLVM: {{.*}} = call double @llvm.fabs.f64(double {{.*}})
// OGCG: {{.*}} = call double @llvm.fabs.f64(double {{.*}})

extern "C" void *test_return_address(void) {
  return __builtin_return_address(1);

  // CIR-LABEL: test_return_address
  // CIR: [[ARG:%.*]] = cir.const #cir.int<1> : !u32i
  // CIR: {{%.*}} = cir.return_address([[ARG]])

  // LLVM-LABEL: @test_return_address
  // LLVM: {{%.*}} = call ptr @llvm.returnaddress(i32 1)

  // OGCG-LABEL: @test_return_address
  // OGCG: {{%.*}} = call ptr @llvm.returnaddress(i32 1)
}

extern "C" void *test_frame_address(void) {
  return __builtin_frame_address(1);

  // CIR-LABEL: test_frame_address
  // CIR: [[ARG:%.*]] = cir.const #cir.int<1> : !u32i
  // CIR: {{%.*}} = cir.frame_address([[ARG]])

  // LLVM-LABEL: @test_frame_address
  // LLVM: {{%.*}} = call ptr @llvm.frameaddress.p0(i32 1)

  // OGCG-LABEL: @test_frame_address
  // OGCG: {{%.*}} = call ptr @llvm.frameaddress.p0(i32 1)
}

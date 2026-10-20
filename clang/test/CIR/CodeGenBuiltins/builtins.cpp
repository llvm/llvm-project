// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -Wno-unused-value -fclangir -emit-cir %s -o %t-amdgpu.cir
// RUN: FileCheck --input-file=%t-amdgpu.cir %s -check-prefix=CIR-AMDGPU
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -Wno-unused-value -fclangir -emit-llvm %s -o %t-amdgpu.ll
// RUN: FileCheck --input-file=%t-amdgpu.ll %s -check-prefix=LLVM-AMDGPU

double fabs(double x) {
  return __builtin_fabs(x);
}

// CIR: {{.*}} = cir.fabs {{.*}} : !cir.double
// LLVM: {{.*}} = call double @llvm.fabs.f64(double {{.*}})

extern "C" void *test_return_address(void) {
  return __builtin_return_address(1);

  // CIR-LABEL: test_return_address
  // CIR: [[ARG:%.*]] = cir.const #cir.int<1> : !u32i
  // CIR: {{%.*}} = cir.return_address([[ARG]])

  // LLVM-LABEL: @test_return_address
  // LLVM: {{%.*}} = call ptr @llvm.returnaddress.p0(i32 1)

}

extern "C" void *test_frame_address(void) {
  return __builtin_frame_address(1);

  // CIR-LABEL: test_frame_address
  // CIR: [[ARG:%.*]] = cir.const #cir.int<1> : !u32i
  // CIR: %[[ADDR:.*]] = cir.frame_address([[ARG]]) : !cir.ptr<!u8i>
  // CIR: {{%.*}} = cir.cast bitcast %[[ADDR]] : !cir.ptr<!u8i> -> !cir.ptr<!void>

  // LLVM-LABEL: @test_frame_address
  // LLVM: {{%.*}} = call ptr @llvm.frameaddress.p0(i32 1)

  // CIR-AMDGPU-LABEL: test_frame_address
  // CIR-AMDGPU: [[ARG:%.*]] = cir.const #cir.int<1> : !u32i
  // CIR-AMDGPU: %[[ADDR:.*]] = cir.frame_address([[ARG]]) : !cir.ptr<!u8i, target_address_space(5)>
  // CIR-AMDGPU: %[[CAST:.*]] = cir.cast bitcast %[[ADDR]] : !cir.ptr<!u8i, target_address_space(5)> -> !cir.ptr<!void, target_address_space(5)>
  // CIR-AMDGPU: {{%.*}} = cir.cast address_space %[[CAST]] : !cir.ptr<!void, target_address_space(5)> -> !cir.ptr<!void>

  // LLVM-AMDGPU-LABEL: @test_frame_address
  // LLVM-AMDGPU: %[[ADDR:.*]] = call ptr addrspace(5) @llvm.frameaddress.p5(i32 1)
  // LLVM-AMDGPU: {{%.*}} = addrspacecast ptr addrspace(5) %[[ADDR]] to ptr

}

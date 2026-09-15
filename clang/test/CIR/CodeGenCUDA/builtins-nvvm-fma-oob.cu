// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -target-cpu sm_90a \
// RUN:            -target-feature +ptx87 -x cuda -fcuda-is-device -fclangir -emit-cir \
// RUN:            %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s

// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -target-cpu sm_90a \
// RUN:            -target-feature +ptx87 -x cuda -fcuda-is-device -fclangir -emit-llvm \
// RUN:            %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s

// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -target-cpu sm_90a \
// RUN:            -target-feature +ptx87 -x cuda -fcuda-is-device -emit-llvm \
// RUN:            %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// FIXME: CIR doesn't propagate the 'contract' fast-math flag to LLVM IR
// calls yet, so LLVM check lines use {{.*}} to tolerate the difference
// between CIR (no flags) and classic codegen ('contract').

#define __device__ __attribute__((device))

typedef __fp16 f16x2 __attribute__((ext_vector_type(2)));
typedef __bf16 bf16x2 __attribute__((ext_vector_type(2)));

// CIR-LABEL: @test_fma_rn_oob_bf16
// CIR: cir.call_llvm_intrinsic "nvvm.fma.rn.oob" {{.*}} : (!cir.bf16, !cir.bf16, !cir.bf16) -> !cir.bf16
// LLVM-LABEL: @test_fma_rn_oob_bf16(
// LLVM: call{{.*}} bfloat @llvm.nvvm.fma.rn.oob.bf16(bfloat %{{.*}}, bfloat %{{.*}}, bfloat %{{.*}})
extern "C" __device__ __bf16 test_fma_rn_oob_bf16(__bf16 a0, __bf16 a1, __bf16 a2) {
  return __nvvm_fma_rn_oob_bf16(a0, a1, a2);
}

// CIR-LABEL: @test_fma_rn_oob_bf16x2
// CIR: cir.call_llvm_intrinsic "nvvm.fma.rn.oob" {{.*}} : (!cir.vector<2 x !cir.bf16>, !cir.vector<2 x !cir.bf16>, !cir.vector<2 x !cir.bf16>) -> !cir.vector<2 x !cir.bf16>
// LLVM-LABEL: @test_fma_rn_oob_bf16x2(
// LLVM: call{{.*}} <2 x bfloat> @llvm.nvvm.fma.rn.oob.v2bf16(<2 x bfloat> %{{.*}}, <2 x bfloat> %{{.*}}, <2 x bfloat> %{{.*}})
extern "C" __device__ bf16x2 test_fma_rn_oob_bf16x2(bf16x2 a0, bf16x2 a1, bf16x2 a2) {
  return __nvvm_fma_rn_oob_bf16x2(a0, a1, a2);
}

// CIR-LABEL: @test_fma_rn_oob_f16
// CIR: cir.call_llvm_intrinsic "nvvm.fma.rn.oob" {{.*}} : (!cir.f16, !cir.f16, !cir.f16) -> !cir.f16
// LLVM-LABEL: @test_fma_rn_oob_f16(
// LLVM: call{{.*}} half @llvm.nvvm.fma.rn.oob.f16(half %{{.*}}, half %{{.*}}, half %{{.*}})
extern "C" __device__ _Float16 test_fma_rn_oob_f16(_Float16 a0, _Float16 a1, _Float16 a2) {
  return __nvvm_fma_rn_oob_f16(a0, a1, a2);
}

// CIR-LABEL: @test_fma_rn_oob_f16x2
// CIR: cir.call_llvm_intrinsic "nvvm.fma.rn.oob" {{.*}} : (!cir.vector<2 x !cir.f16>, !cir.vector<2 x !cir.f16>, !cir.vector<2 x !cir.f16>) -> !cir.vector<2 x !cir.f16>
// LLVM-LABEL: @test_fma_rn_oob_f16x2(
// LLVM: call{{.*}} <2 x half> @llvm.nvvm.fma.rn.oob.v2f16(<2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}})
extern "C" __device__ f16x2 test_fma_rn_oob_f16x2(f16x2 a0, f16x2 a1, f16x2 a2) {
  return __nvvm_fma_rn_oob_f16x2(a0, a1, a2);
}

// CIR-LABEL: @test_fma_rn_oob_relu_bf16
// CIR: cir.call_llvm_intrinsic "nvvm.fma.rn.oob.relu" {{.*}} : (!cir.bf16, !cir.bf16, !cir.bf16) -> !cir.bf16
// LLVM-LABEL: @test_fma_rn_oob_relu_bf16(
// LLVM: call{{.*}} bfloat @llvm.nvvm.fma.rn.oob.relu.bf16(bfloat %{{.*}}, bfloat %{{.*}}, bfloat %{{.*}})
extern "C" __device__ __bf16 test_fma_rn_oob_relu_bf16(__bf16 a0, __bf16 a1, __bf16 a2) {
  return __nvvm_fma_rn_oob_relu_bf16(a0, a1, a2);
}

// CIR-LABEL: @test_fma_rn_oob_relu_bf16x2
// CIR: cir.call_llvm_intrinsic "nvvm.fma.rn.oob.relu" {{.*}} : (!cir.vector<2 x !cir.bf16>, !cir.vector<2 x !cir.bf16>, !cir.vector<2 x !cir.bf16>) -> !cir.vector<2 x !cir.bf16>
// LLVM-LABEL: @test_fma_rn_oob_relu_bf16x2(
// LLVM: call{{.*}} <2 x bfloat> @llvm.nvvm.fma.rn.oob.relu.v2bf16(<2 x bfloat> %{{.*}}, <2 x bfloat> %{{.*}}, <2 x bfloat> %{{.*}})
extern "C" __device__ bf16x2 test_fma_rn_oob_relu_bf16x2(bf16x2 a0, bf16x2 a1, bf16x2 a2) {
  return __nvvm_fma_rn_oob_relu_bf16x2(a0, a1, a2);
}

// CIR-LABEL: @test_fma_rn_oob_relu_f16
// CIR: cir.call_llvm_intrinsic "nvvm.fma.rn.oob.relu" {{.*}} : (!cir.f16, !cir.f16, !cir.f16) -> !cir.f16
// LLVM-LABEL: @test_fma_rn_oob_relu_f16(
// LLVM: call{{.*}} half @llvm.nvvm.fma.rn.oob.relu.f16(half %{{.*}}, half %{{.*}}, half %{{.*}})
extern "C" __device__ _Float16 test_fma_rn_oob_relu_f16(_Float16 a0, _Float16 a1, _Float16 a2) {
  return __nvvm_fma_rn_oob_relu_f16(a0, a1, a2);
}

// CIR-LABEL: @test_fma_rn_oob_relu_f16x2
// CIR: cir.call_llvm_intrinsic "nvvm.fma.rn.oob.relu" {{.*}} : (!cir.vector<2 x !cir.f16>, !cir.vector<2 x !cir.f16>, !cir.vector<2 x !cir.f16>) -> !cir.vector<2 x !cir.f16>
// LLVM-LABEL: @test_fma_rn_oob_relu_f16x2(
// LLVM: call{{.*}} <2 x half> @llvm.nvvm.fma.rn.oob.relu.v2f16(<2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}})
extern "C" __device__ f16x2 test_fma_rn_oob_relu_f16x2(f16x2 a0, f16x2 a1, f16x2 a2) {
  return __nvvm_fma_rn_oob_relu_f16x2(a0, a1, a2);
}

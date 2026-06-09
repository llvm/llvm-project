// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -target-cpu sm_70 \
// RUN:            -target-feature +ptx62 -fclangir -fcuda-is-device \
// RUN:            -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s

// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -target-cpu sm_70 \
// RUN:            -target-feature +ptx62 -fclangir -fcuda-is-device \
// RUN:            -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s

// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -target-cpu sm_70 \
// RUN:            -target-feature +ptx62 -fcuda-is-device \
// RUN:            -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

#define __device__ __attribute__((device))

// Tests CIR/LLVM lowering for NVPTX CTA-level sync barrier builtins.
// Mirrors the relevant slices of clang/test/CodeGen/builtins-nvptx.c and
// clang/test/CodeGen/builtins-nvptx-ptx60.cu.

// CIR-LABEL: cir.func {{.*}} @_Z9nvvm_syncj
// LLVM-LABEL: define{{.*}} void @_Z9nvvm_syncj(
// OGCG-LABEL: define{{.*}} void @_Z9nvvm_syncj(
__device__ void nvvm_sync(unsigned mask) {
  // CIR:  cir.call_llvm_intrinsic "nvvm.barrier.cta.sync.aligned.all"
  // LLVM: call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  // OGCG: call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  __nvvm_bar_sync(0);

  // CIR:  cir.call_llvm_intrinsic "nvvm.barrier.cta.sync.aligned.all"
  // LLVM: call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  // OGCG: call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  __syncthreads();

  // CIR:  cir.call_llvm_intrinsic "nvvm.barrier.cta.sync.all"
  // LLVM: call void @llvm.nvvm.barrier.cta.sync.all(i32 %{{.*}})
  // OGCG: call void @llvm.nvvm.barrier.cta.sync.all(i32 %{{.*}})
  __nvvm_barrier_sync(mask);

  // CIR:  cir.call_llvm_intrinsic "nvvm.barrier.cta.sync.count"
  // LLVM: call void @llvm.nvvm.barrier.cta.sync.count(i32 %{{.*}}, i32 0)
  // OGCG: call void @llvm.nvvm.barrier.cta.sync.count(i32 %{{.*}}, i32 0)
  __nvvm_barrier_sync_cnt(mask, 0);
}

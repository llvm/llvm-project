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

// Tests CIR/LLVM lowering for NVPTX CTA-level sync and bar0 reduction builtins.
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

// CIR-LABEL: cir.func {{.*}} @_Z20nvvm_bar0_reductionsi
// LLVM-LABEL: define{{.*}} i32 @_Z20nvvm_bar0_reductionsi(
// OGCG-LABEL: define{{.*}} i32 @_Z20nvvm_bar0_reductionsi(
__device__ int nvvm_bar0_reductions(int i) {
  int ret = 0;

  // CIR:  %[[NE_AND:.*]] = cir.cmp ne {{.*}} : !s32i
  // CIR:  %[[AND:.*]] = cir.call_llvm_intrinsic "nvvm.barrier.cta.red.and.aligned.all" {{.*}} : (!s32i, !cir.bool) -> !cir.bool
  // CIR:  cir.cast bool_to_int %[[AND]] : !cir.bool -> !s32i
  // LLVM: %[[NE_AND:.*]] = icmp ne i32 %{{.*}}, 0
  // LLVM: %[[AND:.*]] = call i1 @llvm.nvvm.barrier.cta.red.and.aligned.all(i32 0, i1 %[[NE_AND]])
  // LLVM: zext i1 %[[AND]] to i32
  // OGCG: %[[NE_AND:.*]] = icmp ne i32 %{{.*}}, 0
  // OGCG: %[[AND:.*]] = call i1 @llvm.nvvm.barrier.cta.red.and.aligned.all(i32 0, i1 %[[NE_AND]])
  // OGCG: zext i1 %[[AND]] to i32
  ret += __nvvm_bar0_and(i);

  // CIR:  %[[NE_OR:.*]] = cir.cmp ne {{.*}} : !s32i
  // CIR:  %[[OR:.*]] = cir.call_llvm_intrinsic "nvvm.barrier.cta.red.or.aligned.all" {{.*}} : (!s32i, !cir.bool) -> !cir.bool
  // CIR:  cir.cast bool_to_int %[[OR]] : !cir.bool -> !s32i
  // LLVM: %[[NE_OR:.*]] = icmp ne i32 %{{.*}}, 0
  // LLVM: %[[OR:.*]] = call i1 @llvm.nvvm.barrier.cta.red.or.aligned.all(i32 0, i1 %[[NE_OR]])
  // LLVM: zext i1 %[[OR]] to i32
  // OGCG: %[[NE_OR:.*]] = icmp ne i32 %{{.*}}, 0
  // OGCG: %[[OR:.*]] = call i1 @llvm.nvvm.barrier.cta.red.or.aligned.all(i32 0, i1 %[[NE_OR]])
  // OGCG: zext i1 %[[OR]] to i32
  ret += __nvvm_bar0_or(i);

  // CIR:  %[[NE_POPC:.*]] = cir.cmp ne {{.*}} : !s32i
  // CIR:  cir.call_llvm_intrinsic "nvvm.barrier.cta.red.popc.aligned.all" {{.*}} : (!s32i, !cir.bool) -> !s32i
  // LLVM: %[[NE_POPC:.*]] = icmp ne i32 %{{.*}}, 0
  // LLVM: call i32 @llvm.nvvm.barrier.cta.red.popc.aligned.all(i32 0, i1 %[[NE_POPC]])
  // OGCG: %[[NE_POPC:.*]] = icmp ne i32 %{{.*}}, 0
  // OGCG: call i32 @llvm.nvvm.barrier.cta.red.popc.aligned.all(i32 0, i1 %[[NE_POPC]])
  ret += __nvvm_bar0_popc(i);

  return ret;
}

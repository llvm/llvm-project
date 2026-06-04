// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -target-cpu sm_90 \
// RUN:            -target-feature +ptx80 -fclangir -fcuda-is-device \
// RUN:            -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s

// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -target-cpu sm_90 \
// RUN:            -target-feature +ptx80 -fclangir -fcuda-is-device \
// RUN:            -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s

// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -target-cpu sm_90 \
// RUN:            -target-feature +ptx80 -fcuda-is-device \
// RUN:            -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

#define __global__ __attribute__((global))

// Tests CIR/LLVM lowering for sm_90 cluster barrier and fence builtins.
// Mirrors the relevant slice of clang/test/CodeGenCUDA/builtins-sm90.cu.

// CIR-LABEL: cir.func {{.*}} @_Z6kernelv
// LLVM-LABEL: define{{.*}} void @_Z6kernelv(
// OGCG-LABEL: define{{.*}} void @_Z6kernelv(
__global__ void kernel() {
  // CIR:  cir.call_llvm_intrinsic "nvvm.barrier.cluster.arrive"
  // LLVM: call void @llvm.nvvm.barrier.cluster.arrive()
  // OGCG: call void @llvm.nvvm.barrier.cluster.arrive()
  __nvvm_barrier_cluster_arrive();

  // CIR:  cir.call_llvm_intrinsic "nvvm.barrier.cluster.arrive.relaxed"
  // LLVM: call void @llvm.nvvm.barrier.cluster.arrive.relaxed()
  // OGCG: call void @llvm.nvvm.barrier.cluster.arrive.relaxed()
  __nvvm_barrier_cluster_arrive_relaxed();

  // CIR:  cir.call_llvm_intrinsic "nvvm.barrier.cluster.wait"
  // LLVM: call void @llvm.nvvm.barrier.cluster.wait()
  // OGCG: call void @llvm.nvvm.barrier.cluster.wait()
  __nvvm_barrier_cluster_wait();

  // CIR:  cir.call_llvm_intrinsic "nvvm.fence.sc.cluster"
  // LLVM: call void @llvm.nvvm.fence.sc.cluster()
  // OGCG: call void @llvm.nvvm.fence.sc.cluster()
  __nvvm_fence_sc_cluster();
}

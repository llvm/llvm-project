// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -target-feature +ptx80 \
// RUN:            -target-cpu sm_90 -fclangir -emit-cir -fcuda-is-device -target-sdk-version=12.3 \
// RUN:            %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s

// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -target-feature +ptx80 \
// RUN:            -target-cpu sm_90 -fclangir -emit-llvm -fcuda-is-device -target-sdk-version=12.3 \
// RUN:            %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -target-feature +ptx80 \
// RUN:            -target-cpu sm_90 -fclangir -emit-llvm -fcuda-is-device -target-sdk-version=12.3 \
// RUN:            %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCHECK --input-file=%t.ll %s

// CIR-LABEL: _Z6kernelPlPvj(
// LLVM: define{{.*}} void @_Z6kernelPlPvj(
// OGCHECK: define{{.*}} void @_Z6kernelPlPvj(
__attribute__((global)) void kernel(long *out, void *ptr, unsigned u) {
  // CIR: cir.llvm.intrinsic "nvvm.barrier.cluster.arrive"
  // LLVM: call void @llvm.nvvm.barrier.cluster.arrive()
  // OGCHECK: call void @llvm.nvvm.barrier.cluster.arrive()
  __nvvm_barrier_cluster_arrive();

  // CIR: cir.llvm.intrinsic "nvvm.barrier.cluster.arrive.relaxed"
  // LLVM: call void @llvm.nvvm.barrier.cluster.arrive.relaxed()
  // OGCHECK: call void @llvm.nvvm.barrier.cluster.arrive.relaxed()

  __nvvm_barrier_cluster_arrive_relaxed();
  // CIR: cir.llvm.intrinsic "nvvm.barrier.cluster.wait"
  // LLVM: call void @llvm.nvvm.barrier.cluster.wait()
  // OGCHECK: call void @llvm.nvvm.barrier.cluster.wait()
  __nvvm_barrier_cluster_wait();

  // CIR: cir.llvm.intrinsic "nvvm.fence.sc.cluster"
  // LLVM: call void @llvm.nvvm.fence.sc.cluster()
  // OGCHECK: call void @llvm.nvvm.fence.sc.cluster()
  __nvvm_fence_sc_cluster();

  // CIR: cir.return
  // LLVM: ret void
  // OGCHECK: ret void
}

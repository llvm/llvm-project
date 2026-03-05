// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -fclangir -target-cpu sm_70 \
// RUN:            -fcuda-is-device -target-feature +ptx60 \
// RUN:            -emit-cir -o %t.cir -x cuda %s
// RUN: FileCheck -check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -fclangir -target-cpu sm_80 \
// RUN:            -fcuda-is-device -target-feature +ptx65 \
// RUN:            -emit-cir -o %t.cir -x cuda %s
// RUN: FileCheck -check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -fclangir -target-cpu sm_80 \
// RUN:            -fcuda-is-device -target-feature +ptx70 \
// RUN:            -emit-cir -o %t.cir -x cuda %s
// RUN: FileCheck -check-prefix=CIR --input-file=%t.cir %s

// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -fclangir -target-cpu sm_70 \
// RUN:            -fcuda-is-device -target-feature +ptx60 \
// RUN:            -emit-llvm -o %t.ll -x cuda %s
// RUN: FileCheck -check-prefix=LLVM --input-file=%t.ll %s
// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -fclangir -target-cpu sm_80 \
// RUN:            -fcuda-is-device -target-feature +ptx65 \
// RUN:            -emit-llvm -o %t.ll -x cuda %s
// RUN: FileCheck -check-prefix=LLVM --input-file=%t.ll %s
// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -fclangir -target-cpu sm_80 \
// RUN:            -fcuda-is-device -target-feature +ptx70 \
// RUN:            -emit-llvm -o %t.ll -x cuda %s
// RUN: FileCheck -check-prefix=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -target-cpu sm_70 \
// RUN:            -fcuda-is-device -target-feature +ptx60 \
// RUN:            -emit-llvm -o %t_og.ll -x cuda %s
// RUN: FileCheck -check-prefix=OGCHECK --input-file=%t_og.ll %s
// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -target-cpu sm_80 \
// RUN:            -fcuda-is-device -target-feature +ptx65 \
// RUN:            -emit-llvm -o %t_og.ll -x cuda %s
// RUN: FileCheck -check-prefix=OGCHECK --input-file=%t_og.ll %s
// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -target-cpu sm_80 \
// RUN:            -fcuda-is-device -target-feature +ptx70 \
// RUN:            -emit-llvm -o %t_og.ll -x cuda %s
// RUN: FileCheck -check-prefix=OGCHECK --input-file=%t_og.ll %s

#define __device__ __attribute__((device))
#define __global__ __attribute__((global))
#define __shared__ __attribute__((shared))
#define __constant__ __attribute__((constant))

typedef unsigned long long uint64_t;

__device__ void nvvm_sync(unsigned mask, int i, float f, int a, int b,
                          bool pred, uint64_t i64) {

  // CIR: cir.llvm.intrinsic "nvvm.bar.warp.sync" {{.*}} : (!u32i)
  // LLVM: call void @llvm.nvvm.bar.warp.sync(i32
  // OGCHECK: call void @llvm.nvvm.bar.warp.sync(i32
  __nvvm_bar_warp_sync(mask);

  // CIR: cir.llvm.intrinsic "nvvm.barrier.cta.sync.all" {{.*}} : (!u32i)
  // LLVM: call void @llvm.nvvm.barrier.cta.sync.all(i32
  // OGCHECK: call void @llvm.nvvm.barrier.cta.sync.all(i32
  __nvvm_barrier_sync(mask);

  // CIR: cir.llvm.intrinsic "nvvm.barrier.cta.sync.count" {{.*}} : (!u32i, !u32i)
  // LLVM: call void @llvm.nvvm.barrier.cta.sync.count(i32
  // OGCHECK: call void @llvm.nvvm.barrier.cta.sync.count(i32
  __nvvm_barrier_sync_cnt(mask, i);

}

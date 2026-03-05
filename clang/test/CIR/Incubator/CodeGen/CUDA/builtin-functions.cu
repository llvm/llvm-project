#include "../Inputs/cuda.h"

// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -fclangir \
// RUN:            -fcuda-is-device -emit-cir -target-sdk-version=12.3 \
// RUN:            %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s

// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -fclangir \
// RUN:            -fcuda-is-device -emit-llvm -target-sdk-version=12.3 \
// RUN:            %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda \
// RUN:            -fcuda-is-device -emit-llvm -target-sdk-version=12.3 \
// RUN:            %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCHECK --input-file=%t.ll %s

__device__ void sync() {

  // CIR: cir.llvm.intrinsic "nvvm.barrier.cta.sync.aligned.all" {{.*}} : (!s32i)
  // LLVM: call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  // OGCHECK: call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  __nvvm_bar_sync(0);
}

__device__ void builtins() {
  float f1, f2;
  double d1, d2;

  // CIR: cir.llvm.intrinsic "nvvm.fmax.f" {{.*}} : (!cir.float, !cir.float) -> !cir.float
  // LLVM: call float @llvm.nvvm.fmax.f(float {{.*}}, float {{.*}})
  float t1 = __nvvm_fmax_f(f1, f2);
  // CIR: cir.llvm.intrinsic "nvvm.fmin.f" {{.*}} : (!cir.float, !cir.float) -> !cir.float
  // LLVM: call float @llvm.nvvm.fmin.f(float {{.*}}, float {{.*}})
  float t2 = __nvvm_fmin_f(f1, f2);
  // CIR: cir.llvm.intrinsic "nvvm.sqrt.rn.f" {{.*}} : (!cir.float) -> !cir.float
  // LLVM: call float @llvm.nvvm.sqrt.rn.f(float {{.*}})
  float t3 = __nvvm_sqrt_rn_f(f1);
  // CIR: cir.llvm.intrinsic "nvvm.rcp.rn.f" {{.*}} : (!cir.float) -> !cir.float
  // LLVM: call float @llvm.nvvm.rcp.rn.f(float {{.*}})
  float t4 = __nvvm_rcp_rn_f(f2);
  // CIR: cir.llvm.intrinsic "nvvm.add.rn.f" {{.*}} : (!cir.float, !cir.float) -> !cir.float
  // LLVM: call float @llvm.nvvm.add.rn.f(float {{.*}}, float {{.*}})
  float t5 = __nvvm_add_rn_f(f1, f2);

  // CIR: cir.llvm.intrinsic "nvvm.fmax.d" {{.*}} : (!cir.double, !cir.double) -> !cir.double
  // LLVM: call double @llvm.nvvm.fmax.d(double {{.*}}, double {{.*}})
  double td1 = __nvvm_fmax_d(d1, d2);
  // CIR: cir.llvm.intrinsic "nvvm.fmin.d" {{.*}} : (!cir.double, !cir.double) -> !cir.double
  // LLVM: call double @llvm.nvvm.fmin.d(double {{.*}}, double {{.*}})
  double td2 = __nvvm_fmin_d(d1, d2);
  // CIR: cir.llvm.intrinsic "nvvm.sqrt.rn.d" {{.*}} : (!cir.double) -> !cir.double
  // LLVM: call double @llvm.nvvm.sqrt.rn.d(double {{.*}})
  double td3 = __nvvm_sqrt_rn_d(d1);
  // CIR: cir.llvm.intrinsic "nvvm.rcp.rn.d" {{.*}} : (!cir.double) -> !cir.double
  // LLVM: call double @llvm.nvvm.rcp.rn.d(double {{.*}})
  double td4 = __nvvm_rcp_rn_d(d2);

  int i1, i2;

  // CIR: cir.llvm.intrinsic "nvvm.mulhi.i" {{.*}} : (!s32i, !s32i) -> !s32i
  // LLVM: call i32 @llvm.nvvm.mulhi.i(i32 {{.*}}, i32 {{.*}})
  int ti1 = __nvvm_mulhi_i(i1, i2);

  // CIR: cir.llvm.intrinsic "nvvm.membar.cta"
  // LLVM: call void @llvm.nvvm.membar.cta()
  __nvvm_membar_cta();
  // CIR: cir.llvm.intrinsic "nvvm.membar.gl"
  // LLVM: call void @llvm.nvvm.membar.gl()
  __nvvm_membar_gl();
  // CIR: cir.llvm.intrinsic "nvvm.membar.sys"
  // LLVM: call void @llvm.nvvm.membar.sys()
  __nvvm_membar_sys();
  
  // CIR: cir.llvm.intrinsic "nvvm.barrier.cta.sync.aligned.all"
  // LLVM: call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  // OGCHECK: call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  __syncthreads();
}

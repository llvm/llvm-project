// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -fclangir \
// RUN:            -fcuda-is-device -emit-llvm -o - %s   \
// RUN: | FileCheck --check-prefix=LLVM %s

// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -fclangir \
// RUN:            -fcuda-is-device -emit-cir -o - %s   \
// RUN: | FileCheck --check-prefix=CIR %s

// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda  \
// RUN:            -fcuda-is-device -emit-llvm -o - %s   \
// RUN: | FileCheck --check-prefix=OGCG %s

#include "__clang_cuda_builtin_vars.h"

// LLVM: define{{.*}} void @_Z6kernelPi(ptr %0)
// OGCG: define{{.*}} void @_Z6kernelPi(ptr noundef %out)
__attribute__((global))
void kernel(int *out) {
  int i = 0;

  out[i++] = threadIdx.x;
  // CIR:  cir.func {{.*}} @_ZN26__cuda_builtin_threadIdx_t17__fetch_builtin_xEv()
  // CIR:  cir.llvm.intrinsic "nvvm.read.ptx.sreg.tid.x"
  // LLVM: call{{.*}} i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  // OGCG: call noundef{{.*}} i32 @llvm.nvvm.read.ptx.sreg.tid.x()

  out[i++] = threadIdx.y;
  // CIR:  cir.func {{.*}} @_ZN26__cuda_builtin_threadIdx_t17__fetch_builtin_yEv()
  // CIR:  cir.llvm.intrinsic "nvvm.read.ptx.sreg.tid.y"
  // LLVM: call{{.*}} i32 @llvm.nvvm.read.ptx.sreg.tid.y()
  // OGCG: call noundef{{.*}} i32 @llvm.nvvm.read.ptx.sreg.tid.y()

  out[i++] = threadIdx.z;
  // CIR:  cir.func {{.*}} @_ZN26__cuda_builtin_threadIdx_t17__fetch_builtin_zEv()
  // CIR:  cir.llvm.intrinsic "nvvm.read.ptx.sreg.tid.z"
  // LLVM: call{{.*}} i32 @llvm.nvvm.read.ptx.sreg.tid.z()
  // OGCG: call noundef{{.*}} i32 @llvm.nvvm.read.ptx.sreg.tid.z()


  out[i++] = blockIdx.x;
  // CIR:  cir.func {{.*}} @_ZN25__cuda_builtin_blockIdx_t17__fetch_builtin_xEv()
  // CIR:  cir.llvm.intrinsic "nvvm.read.ptx.sreg.ctaid.x"
  // LLVM: call{{.*}} i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  // OGCG: call noundef{{.*}} i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()

  out[i++] = blockIdx.y;
  // CIR:  cir.func {{.*}} @_ZN25__cuda_builtin_blockIdx_t17__fetch_builtin_yEv()
  // CIR:  cir.llvm.intrinsic "nvvm.read.ptx.sreg.ctaid.y"
  // LLVM: call{{.*}} i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()
  // OGCG: call noundef{{.*}} i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()

  out[i++] = blockIdx.z;
  // CIR:  cir.func {{.*}} @_ZN25__cuda_builtin_blockIdx_t17__fetch_builtin_zEv()
  // CIR:  cir.llvm.intrinsic "nvvm.read.ptx.sreg.ctaid.z"
  // LLVM: call{{.*}} i32 @llvm.nvvm.read.ptx.sreg.ctaid.z()
  // OGCG: call noundef{{.*}} i32 @llvm.nvvm.read.ptx.sreg.ctaid.z()


  out[i++] = blockDim.x;
  // CIR:  cir.func {{.*}} @_ZN25__cuda_builtin_blockDim_t17__fetch_builtin_xEv()
  // CIR:  cir.llvm.intrinsic "nvvm.read.ptx.sreg.ntid.x"
  // LLVM: call{{.*}} i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  // OGCG: call noundef{{.*}} i32 @llvm.nvvm.read.ptx.sreg.ntid.x()

  out[i++] = blockDim.y;
  // CIR:  cir.func {{.*}} @_ZN25__cuda_builtin_blockDim_t17__fetch_builtin_yEv()
  // CIR:  cir.llvm.intrinsic "nvvm.read.ptx.sreg.ntid.y"
  // LLVM: call{{.*}} i32 @llvm.nvvm.read.ptx.sreg.ntid.y()
  // OGCG: call noundef{{.*}} i32 @llvm.nvvm.read.ptx.sreg.ntid.y()

  out[i++] = blockDim.z;
  // CIR:  cir.func {{.*}} @_ZN25__cuda_builtin_blockDim_t17__fetch_builtin_zEv()
  // CIR:  cir.llvm.intrinsic "nvvm.read.ptx.sreg.ntid.z"
  // LLVM: call{{.*}} i32 @llvm.nvvm.read.ptx.sreg.ntid.z()
  // OGCG: call noundef{{.*}} i32 @llvm.nvvm.read.ptx.sreg.ntid.z()


  out[i++] = gridDim.x;
  // CIR:  cir.func {{.*}} @_ZN24__cuda_builtin_gridDim_t17__fetch_builtin_xEv()
  // CIR:  cir.llvm.intrinsic "nvvm.read.ptx.sreg.nctaid.x"
  // LLVM: call{{.*}} i32 @llvm.nvvm.read.ptx.sreg.nctaid.x()
  // OGCG: call noundef{{.*}} i32 @llvm.nvvm.read.ptx.sreg.nctaid.x()

  out[i++] = gridDim.y;
  // CIR:  cir.func {{.*}} @_ZN24__cuda_builtin_gridDim_t17__fetch_builtin_yEv()
  // CIR:  cir.llvm.intrinsic "nvvm.read.ptx.sreg.nctaid.y"
  // LLVM: call{{.*}} i32 @llvm.nvvm.read.ptx.sreg.nctaid.y()
  // OGCG: call noundef{{.*}} i32 @llvm.nvvm.read.ptx.sreg.nctaid.y()

  out[i++] = gridDim.z;
  // CIR:  cir.func {{.*}} @_ZN24__cuda_builtin_gridDim_t17__fetch_builtin_zEv()
  // CIR:  cir.llvm.intrinsic "nvvm.read.ptx.sreg.nctaid.z"
  // LLVM: call{{.*}} i32 @llvm.nvvm.read.ptx.sreg.nctaid.z()
  // OGCG: call noundef{{.*}} i32 @llvm.nvvm.read.ptx.sreg.nctaid.z()


  out[i++] = warpSize;
  // CIR: [[REGISTER:%.*]] = cir.const #cir.int<32>
  // CIR: cir.store{{.*}} [[REGISTER]]
  // LLVM: store i32 32,
  // OGCG: store i32 32,


  // CIR: cir.return loc
  // LLVM: ret void
  // OGCG: ret void
}

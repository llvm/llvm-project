// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple spirv64-amd-amdhsa -x hip -fclangir \
// RUN:            -fcuda-is-device -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR %s --input-file=%t.cir

// RUN: %clang_cc1 -triple spirv64-amd-amdhsa -x hip -fclangir \
// RUN:            -fcuda-is-device -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM %s --input-file=%t-cir.ll

// RUN: %clang_cc1 -triple spirv64-amd-amdhsa -x hip \
// RUN:            -fcuda-is-device -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG %s --input-file=%t.ll

// Test that AMDGPU builtins are available on AMDGCN-flavored SPIR-V.

#define __device__ __attribute__((device))

__device__ int test_readfirstlane(int x) {
  return __builtin_amdgcn_readfirstlane(x);
}

// CIR-LABEL: cir.func no_inline @_Z18test_readfirstlanei
// CIR: cir.call_llvm_intrinsic "amdgcn.readfirstlane" {{.*}} : (!s32i) -> !s32i

// LLVM-LABEL: define spir_func noundef i32 @_Z18test_readfirstlanei
// LLVM: call{{.*}} @llvm.amdgcn.readfirstlane.i32

// OGCG-LABEL: define spir_func noundef i32 @_Z18test_readfirstlanei
// OGCG: addrspacecast ptr %{{.*}} to ptr addrspace(4)
// OGCG: call{{.*}} @llvm.amdgcn.readfirstlane.i32

__device__ float test_rcp(float x) {
  return __builtin_amdgcn_rcpf(x);
}

// CIR-LABEL: cir.func no_inline @_Z8test_rcpf
// CIR: cir.call_llvm_intrinsic "amdgcn.rcp" {{.*}} : (!cir.float) -> !cir.float

// LLVM-LABEL: define spir_func noundef float @_Z8test_rcpf
// LLVM: call{{.*}} @llvm.amdgcn.rcp.f32

// OGCG-LABEL: define spir_func noundef float @_Z8test_rcpf
// OGCG: call contract{{.*}} @llvm.amdgcn.rcp.f32

// Reached through the generic clang-builtin-to-intrinsic mapping, which has to
// retry the "amdgcn" prefix after "spv" fails to match.

__device__ unsigned test_wavefrontsize() {
  return __builtin_amdgcn_wavefrontsize();
}

// CIR-LABEL: cir.func no_inline @_Z18test_wavefrontsizev
// CIR: cir.call_llvm_intrinsic "amdgcn.wavefrontsize" : () -> !u32i

// LLVM-LABEL: define spir_func noundef i32 @_Z18test_wavefrontsizev
// LLVM: call{{.*}} @llvm.amdgcn.wavefrontsize()

// OGCG-LABEL: define spir_func noundef i32 @_Z18test_wavefrontsizev
// OGCG: call{{.*}} @llvm.amdgcn.wavefrontsize()

// Expanded inline rather than emitted as a libm call, because SPIR-V with an
// AMD vendor has no device libm.

__device__ float test_logb(float x) {
  return __builtin_logbf(x);
}

// CIR-LABEL: cir.func no_inline @_Z9test_logbf
// CIR: cir.call_llvm_intrinsic "frexp" %{{.*}} : (!cir.float) -> !rec_anon_struct

// LLVM-LABEL: define spir_func noundef float @_Z9test_logbf
// LLVM: call{{.*}} @llvm.frexp.f32.i32
// LLVM: add i32 %{{.*}}, -1

// OGCG-LABEL: define spir_func noundef float @_Z9test_logbf
// OGCG: call{{.*}} @llvm.frexp.f32.i32
// OGCG: add nsw i32 %{{.*}}, -1
// OGCG: load float, ptr addrspace(4) %{{.*}}
// OGCG: call contract{{.*}} @llvm.fabs.f32

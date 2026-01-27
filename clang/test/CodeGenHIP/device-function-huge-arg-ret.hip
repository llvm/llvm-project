// RUN: %clang_cc1 -triple spirv64-amd-amdhsa -x hip -emit-llvm -fcuda-is-device -disable-llvm-verifier -o - %s | FileCheck %s --check-prefix=CHECK-AMDGCNSPIRV
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -x hip -emit-llvm -fcuda-is-device -disable-llvm-verifier -o - %s | FileCheck %s --check-prefix=CHECK-AMDGCN

// NOTE: The verifier is currently disabled as it complains about the 'byref'
//       arguments being too large. This is currently a problem for all targets
//       that lower large arguments to 'byref' arguments.

#define __device__ __attribute__((device))
#define __global__ __attribute__((global))

typedef struct {
  long data[6871947673600];
} huge_struct;

// CHECK-AMDGCNSPIRV: define spir_func void @_Z21deviceFuncWithHugeRetv(ptr dead_on_unwind noalias writable sret(%struct.huge_struct) align 8
// CHECK-AMDGCN: define dso_local void @_Z21deviceFuncWithHugeRetv(ptr addrspace(5) dead_on_unwind noalias writable sret(%struct.huge_struct) align 8
__device__ huge_struct deviceFuncWithHugeRet() { return {}; }

// CHECK-AMDGCNSPIRV: define spir_func void @_Z21deviceFuncWithHugeArg11huge_struct(ptr noundef byref(%struct.huge_struct) align 8
// CHECK-AMDGCN: define dso_local void @_Z21deviceFuncWithHugeArg11huge_struct(ptr addrspace(5) noundef byref(%struct.huge_struct) align 8
__device__ void deviceFuncWithHugeArg(huge_struct X) {}

__device__ void deviceCaller() {
  // CHECK-AMDGCNSPIRV: call spir_func addrspace(4) void @_Z21deviceFuncWithHugeRetv(ptr dead_on_unwind writable sret(%struct.huge_struct) align 8
  // CHECK-AMDGCN: call void @_Z21deviceFuncWithHugeRetv(ptr addrspace(5) dead_on_unwind writable sret(%struct.huge_struct) align 8
  huge_struct X = deviceFuncWithHugeRet();
  // CHECK-AMDGCNSPIRV: call spir_func addrspace(4) void @_Z21deviceFuncWithHugeArg11huge_struct(ptr noundef byref(%struct.huge_struct) align 8
  // CHECK-AMDGCN: call void @_Z21deviceFuncWithHugeArg11huge_struct(ptr addrspace(5) noundef byref(%struct.huge_struct) align 8
  deviceFuncWithHugeArg(X);
}

// CHECK-AMDGCNSPIRV: define spir_kernel void @_Z21globalFuncWithHugeArg11huge_struct(ptr addrspace(2) noundef byref(%struct.huge_struct) align 8
// CHECK-AMDGCN: define dso_local amdgpu_kernel void @_Z21globalFuncWithHugeArg11huge_struct(ptr addrspace(4) noundef byref(%struct.huge_struct) align 8
__global__ void globalFuncWithHugeArg(huge_struct X) {}

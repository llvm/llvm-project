// RUN: %clang_cc1 -triple spirv64-amd-amdhsa -x hip -emit-llvm -fcuda-is-device -disable-llvm-verifier -o - %s | FileCheck %s --check-prefix=CHECK-AMDGCNSPIRV
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -x hip -emit-llvm -fcuda-is-device -o - %s | FileCheck %s --check-prefix=CHECK-AMDGCN

// NOTE: The verifier is currently disabled for the spirv64 case as it complains
//       about the 'byref' arguments being too large. This is currently a
//       problem for all targets that lower large arguments to 'byref'
//       arguments.

#define __device__ __attribute__((device))

typedef struct {
  long data[6871947673600];
} huge_struct;

// CHECK-AMDGCNSPIRV: @_Z9printBits11huge_struct(ptr noundef byref(%struct.huge_struct)
// CHECK-AMDGCN: @_Z9printBits11huge_struct(i16
__device__ void printBits(huge_struct X) {}

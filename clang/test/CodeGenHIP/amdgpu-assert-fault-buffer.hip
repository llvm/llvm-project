// Create module flag for assert fault buffer.

// RUN: %clang_cc1 -x hip -fcuda-is-device -triple amdgcn-amd-amdhsa -emit-llvm \
// RUN: -o - %s | FileCheck %s -check-prefix=NOFLAG

// RUN: %clang_cc1 -x hip -fcuda-is-device -triple amdgcn-amd-amdhsa -emit-llvm \
// RUN: -mamdgpu-assert-fault-buffer -o - %s | FileCheck %s -check-prefix=FLAG

// NOFLAG-NOT: amdgpu_assert_fault_buffer
// FLAG: !{{.*}} = !{i32 1, !"amdgpu_assert_fault_buffer", i32 1}

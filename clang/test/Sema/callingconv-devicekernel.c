// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -emit-llvm %s 2>&1 -o -| FileCheck -check-prefix=CHECK-AMDGPU %s
// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda- -emit-llvm %s 2>&1 -o -| FileCheck -check-prefix=CHECK-NVPTX %s
// RUN: %clang_cc1 -triple spir64 -emit-llvm %s 2>&1 -o - | FileCheck -check-prefix=CHECK-SPIR %s
// RUN: %clang_cc1 -triple spirv64 -emit-llvm %s 2>&1 -o - | FileCheck -check-prefix=CHECK-SPIR %s

// CHECK-AMDGPU-DAG: amdgpu_kernel void @kernel1()
// CHECK-NVPTX-DAG: ptx_kernel void @kernel1()
// CHECK-SPIR-DAG: spir_kernel void @kernel1()
[[clang::device_kernel]] void kernel1() {}

// CHECK-AMDGPU-DAG: amdgpu_kernel void @kernel2()
// CHECK-NVPTX-DAG: 14:3: warning: 'clang::amdgpu_kernel' calling convention is not supported for this target
// CHECK-SPIR-DAG: 14:3: warning: 'clang::amdgpu_kernel' calling convention is not supported for this target
[[clang::amdgpu_kernel]] void kernel2() {}

// CHECK-AMDGPU-DAG: 19:3: warning: 'clang::nvptx_kernel' calling convention is not supported for this target
// CHECK-NVPTX-DAG: ptx_kernel void @kernel3()
// CHECK-SPIR-DAG: 19:3: warning: 'clang::nvptx_kernel' calling convention is not supported for this target
[[clang::nvptx_kernel]] void kernel3() {}

// CHECK-AMDGPU-DAG: 24:3: warning: 'clang::sycl_kernel' attribute ignored
// CHECK-NVPTX-DAG: 24:3: warning: 'clang::sycl_kernel' attribute ignored
// CHECK-SPIR-DAG: 24:3: warning: 'clang::sycl_kernel' attribute ignored
[[clang::sycl_kernel]] void kernel4() {}

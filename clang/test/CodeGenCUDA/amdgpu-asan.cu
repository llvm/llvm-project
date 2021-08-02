// RUN: %clang_cc1 %s -emit-llvm -o - -triple=amdgcn-amd-amdhsa \
// RUN:   -fcuda-is-device -target-cpu gfx906 -fsanitize=address \
// RUN:   -x hip | FileCheck -check-prefix=ASAN %s

// RUN: %clang_cc1 %s -emit-llvm -o - -triple=amdgcn-amd-amdhsa \
// RUN:   -fcuda-is-device -target-cpu gfx906 -x hip \
// RUN:   | FileCheck %s

// REQUIRES: amdgpu-registered-target

// ASAN-DAG: declare void @__amdgpu_device_library_preserve_asan_functions()
// ASAN-DAG: @__amdgpu_device_library_preserve_asan_functions_ptr = weak addrspace(1) constant void ()* @__amdgpu_device_library_preserve_asan_functions
// ASAN-DAG: @llvm.compiler.used = {{.*}}@__amdgpu_device_library_preserve_asan_functions_ptr

// CHECK-NOT: @__amdgpu_device_library_preserve_asan_functions_ptr

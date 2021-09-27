// Create a sample address sanitizer bitcode library.

// RUN: %clang_cc1 -x ir -fcuda-is-device -triple amdgcn-amd-amdhsa -emit-llvm-bc \
// RUN:   -disable-llvm-passes -o %t.asanrtl.bc %S/Inputs/amdgpu-asanrtl.ll

// Check sanitizer runtime library functions survive
// optimizations without being removed or parameters altered.

// RUN: %clang_cc1 %s -emit-llvm -o - -triple=amdgcn-amd-amdhsa \
// RUN:   -fcuda-is-device -target-cpu gfx906 -fsanitize=address \
// RUN:   -mlink-bitcode-file %t.asanrtl.bc -x hip \
// RUN:   | FileCheck -check-prefix=ASAN %s

// RUN: %clang_cc1 %s -emit-llvm -o - -triple=amdgcn-amd-amdhsa \
// RUN:   -fcuda-is-device -target-cpu gfx906 -fsanitize=address \
// RUN:   -O3 -mlink-bitcode-file %t.asanrtl.bc -x hip \
// RUN:   | FileCheck -check-prefix=ASAN %s

// RUN: %clang_cc1 %s -emit-llvm -o - -triple=amdgcn-amd-amdhsa \
// RUN:   -fcuda-is-device -target-cpu gfx906 -x hip \
// RUN:   | FileCheck %s

// REQUIRES: amdgpu-registered-target

// ASAN-DAG: define weak void @__amdgpu_device_library_preserve_asan_functions()
// ASAN-DAG: @__amdgpu_device_library_preserve_asan_functions_ptr = weak addrspace(1) constant void ()* @__amdgpu_device_library_preserve_asan_functions
// ASAN-DAG: @llvm.compiler.used = {{.*}}@__amdgpu_device_library_preserve_asan_functions_ptr
// ASAN-DAG: define weak void @__asan_report_load1(i64 %{{.*}})

// CHECK-NOT: @__amdgpu_device_library_preserve_asan_functions
// CHECK-NOT: @__asan_report_load1

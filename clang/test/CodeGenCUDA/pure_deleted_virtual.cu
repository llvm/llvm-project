// RUN: %clang --cuda-device-only -S -emit-llvm -o - %s 2>&1 | FileCheck %s

// Check that __cxa_pure_virtual() and __cxa_deleted_virtual() are always
// available in device code. These functions are defined in a header included
// by __clang_cuda_runtime_wrapper.h, so use the driver here rather than
// invoking the frontend directly to make sure they are pulled in.

// CHECK-DAG: define weak {{.*}} void @__cxa_pure_virtual()
// CHECK-DAG: define weak {{.*}} void @__cxa_deleted_virtual()

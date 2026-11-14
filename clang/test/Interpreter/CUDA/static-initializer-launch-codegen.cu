// REQUIRES: lld
// Tests a kernel launch from the input that defines the kernel
// RUN: rm -rf %t
// RUN: mkdir -p %t/lib
// RUN: %clang --target=%host-jit-triple -nostdlib -fPIC -shared -fuse-ld=lld \
// RUN:   %S/Inputs/cudart.c -o %t/lib/libcudart.so
// RUN: cat %s | clang-repl --cuda --cuda-path=%t -Xcc -nocudainc \
// RUN:   -Xcc -nocudalib -Xcc -include -Xcc %S/Inputs/cuda.h \
// RUN:   -Xcc -Xclang -Xcc -emit-llvm | FileCheck %s

__global__ void test_func() {} \
test_func<<<1,1>>>();
// The device binary is registered before the input's static initializer
// launches the kernel.
// CHECK: define {{.*}}ptx_kernel void @_Z9test_funcv(
// CHECK: @llvm.global_ctors = appending global [2 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @__cuda_module_ctor, ptr null }, { i32, ptr, ptr } { i32 65535, ptr @_GLOBAL__sub_I_{{[^,]+}}, ptr null }]

%quit

// RUN: rm -rf %t
// RUN: mkdir -p %t/lib
// RUN: %clang --target=%host-jit-triple -nostdlib -fPIC -shared \
// RUN:   %S/Inputs/cudart.c -o %t/lib/libcudart.so
// RUN: cat %s | clang-repl --cuda --cuda-path=%t -Xcc -nocudainc \
// RUN:   -Xcc -nocudalib -Xcc -include -Xcc %S/Inputs/cuda.h \
// RUN:   -Xcc -Xclang -Xcc -emit-llvm | FileCheck %s

__global__ void test_func() {}
// CHECK: define {{.*}}ptx_kernel void @_Z9test_funcv(
// CHECK: define {{.*}}void @_Z24__device_stub__test_funcv(
// CHECK: call i32 @cudaLaunch(ptr @_Z24__device_stub__test_funcv)
// CHECK: call i32 @__cudaRegisterFunction({{.*}}@_Z24__device_stub__test_funcv
// CHECK: call ptr @__cudaRegisterFatBinary(

test_func<<<1,1>>>();
// CHECK: call i32 @cudaConfigureCall(
// CHECK: call void @_Z24__device_stub__test_funcv()

%quit

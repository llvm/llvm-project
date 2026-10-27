// Tests __device__ function calls
// RUN: rm -rf %t
// RUN: mkdir -p %t/lib
// RUN: %clang --target=%host-jit-triple -nostdlib -fPIC -shared \
// RUN:   %S/Inputs/cudart.c -o %t/lib/libcudart.so
// RUN: cat %s | clang-repl --cuda --cuda-path=%t -Xcc -nocudainc \
// RUN:   -Xcc -nocudalib -Xcc -include -Xcc %S/Inputs/cuda.h \
// RUN:   -Xcc -Xclang -Xcc -emit-llvm | FileCheck %s

__device__ inline void test_device(int* value) { *value = 42; }
__global__ void test_kernel(int* value) { test_device(value); }
// CHECK: define {{.*}}ptx_kernel void @_Z11test_kernelPi(
// CHECK: call void @_Z11test_devicePi(
// CHECK: define linkonce_odr void @_Z11test_devicePi(
// CHECK: store i32 42
// CHECK: define {{.*}}void @_Z26__device_stub__test_kernelPi(

int* devptr = nullptr;
cudaMalloc((void **) &devptr, sizeof(int));
test_kernel<<<1,1>>>(devptr);
// CHECK: call i32 @cudaConfigureCall(
// CHECK: call void @_Z26__device_stub__test_kernelPi(

%quit

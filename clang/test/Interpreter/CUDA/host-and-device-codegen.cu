// Checks that a function is available in both __host__ and __device__
// RUN: rm -rf %t
// RUN: mkdir -p %t/lib
// RUN: %clang --target=%host-jit-triple -nostdlib -fPIC -shared \
// RUN:   %S/Inputs/cudart.c -o %t/lib/libcudart.so
// RUN: cat %s | clang-repl --cuda --cuda-path=%t -Xcc -nocudainc \
// RUN:   -Xcc -nocudalib -Xcc -include -Xcc %S/Inputs/cuda.h \
// RUN:   -Xcc -Xclang -Xcc -emit-llvm | FileCheck %s

__host__ __device__ inline int sum(int a, int b){ return a + b; }
__global__ void kernel(int * output){ *output = sum(40,2); }
// CHECK: define {{.*}}ptx_kernel void @_Z6kernelPi(
// CHECK: call {{.*}}i32 @_Z3sumii(i32 noundef 40, i32 noundef 2)
// CHECK: define linkonce_odr {{.*}}i32 @_Z3sumii(
// CHECK: define {{.*}}void @_Z21__device_stub__kernelPi(

int host_sum = sum(41, 1);
// CHECK: call {{.*}}i32 @_Z3sumii(i32 noundef 41, i32 noundef 1)
// CHECK: define linkonce_odr {{.*}}i32 @_Z3sumii(

int * deviceVar;
cudaMalloc((void **) &deviceVar, sizeof(int));
kernel<<<1,1>>>(deviceVar);
// CHECK: call i32 @cudaConfigureCall(
// CHECK: call void @_Z21__device_stub__kernelPi(

%quit

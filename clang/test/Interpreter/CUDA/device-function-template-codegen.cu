// Tests device function templates
// RUN: rm -rf %t
// RUN: mkdir -p %t/lib
// RUN: %clang --target=%host-jit-triple -nostdlib -fPIC -shared \
// RUN:   %S/Inputs/cudart.c -o %t/lib/libcudart.so
// RUN: cat %s | clang-repl --cuda --cuda-path=%t -Xcc -nocudainc \
// RUN:   -Xcc -nocudalib -Xcc -include -Xcc %S/Inputs/cuda.h \
// RUN:   -Xcc -Xclang -Xcc -emit-llvm | FileCheck %s

template <typename T> __device__ inline T sum(T a, T b) { return a + b; }
__global__ void test_kernel(int* value) { *value = sum(40, 2); }
// CHECK: define {{.*}}ptx_kernel void @_Z11test_kernelPi(
// CHECK: call {{.*}}i32 @_Z3sumIiET_S0_S0_(i32 noundef 40, i32 noundef 2)
// CHECK: define linkonce_odr {{.*}}i32 @_Z3sumIiET_S0_S0_(
// CHECK: define {{.*}}void @_Z26__device_stub__test_kernelPi(

int* devptr = nullptr;
cudaMalloc((void **) &devptr, sizeof(int));
test_kernel<<<1,1>>>(devptr);
// CHECK: call i32 @cudaConfigureCall(
// CHECK: call void @_Z26__device_stub__test_kernelPi(

%quit

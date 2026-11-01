// Tests cudaMemcpy and writes from kernel
// RUN: rm -rf %t
// RUN: mkdir -p %t/lib
// RUN: %clang --target=%host-jit-triple -nostdlib -fPIC -shared \
// RUN:   %S/Inputs/cudart.c -o %t/lib/libcudart.so
// RUN: cat %s | clang-repl --cuda --cuda-path=%t -Xcc -nocudainc \
// RUN:   -Xcc -nocudalib -Xcc -include -Xcc %S/Inputs/cuda.h \
// RUN:   -Xcc -Xclang -Xcc -emit-llvm | FileCheck %s

__global__ void test_func(int* value) { *value = 42; }
// CHECK: define {{.*}}ptx_kernel void @_Z9test_funcPi(
// CHECK: store i32 42
// CHECK: define {{.*}}void @_Z24__device_stub__test_funcPi(
// CHECK: call i32 @cudaSetupArgument(
// CHECK: call i32 @cudaLaunch(ptr @_Z24__device_stub__test_funcPi)

int var;
int* devptr = nullptr;
cudaMalloc((void **) &devptr, sizeof(int));
// CHECK: call {{.*}}@cudaMalloc(

test_func<<<1,1>>>(devptr);
// CHECK: call i32 @cudaConfigureCall(
// CHECK: call void @_Z24__device_stub__test_funcPi(

cudaMemcpy(&var, devptr, sizeof(int), cudaMemcpyDeviceToHost);
// CHECK: call {{.*}}@cudaMemcpy(

%quit

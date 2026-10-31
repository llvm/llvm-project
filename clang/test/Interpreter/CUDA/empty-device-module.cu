// Tests host-only inputs. They produce an empty device module, and emitting
// PTX for it must not be reported as a failure just because no pass changed
// the module.
// RUN: rm -rf %t
// RUN: mkdir -p %t/lib
// RUN: %clang --target=%host-jit-triple -nostdlib -fPIC -shared \
// RUN:   %S/Inputs/cudart.c -o %t/lib/libcudart.so
// RUN: cat %s | clang-repl --cuda --cuda-path=%t -Xcc -nocudainc \
// RUN:   -Xcc -nocudalib -Xcc -include -Xcc %S/Inputs/cuda.h \
// RUN:   -Xcc -Xclang -Xcc -emit-llvm | FileCheck %s

int host_only = 42;
// CHECK-NOT: define
// CHECK: @host_only = {{.*}}global i32 42

__global__ void kernel() {}
// CHECK: define {{.*}}ptx_kernel void @_Z6kernelv(
// CHECK: call i32 @__cudaRegisterFunction({{.*}}@_Z21__device_stub__kernelv

kernel<<<1,1>>>();
// CHECK: call void @_Z21__device_stub__kernelv()

%quit

// RUN: x=$(%clangxx -x hip --offload-arch=gfx908 -S -emit-llvm -fgpu-rdc %s -o - | md5sum | awk '{ print $1 }') && echo $x > %t.md5
// RUN: y1=$(%clangxx -x hip --offload-arch=gfx908 -S -emit-llvm -fgpu-rdc %s -o - | md5sum | awk '{ print $1 }') && echo $y1 >> %t.md5
// RUN: y2=$(%clangxx -x hip --offload-arch=gfx908 -S -emit-llvm -fgpu-rdc %s -o - | md5sum | awk '{ print $1 }') && echo $y2 >> %t.md5
// RUN: y3=$(%clangxx -x hip --offload-arch=gfx908 -S -emit-llvm -fgpu-rdc %s -o - | md5sum | awk '{ print $1 }') && echo $y3 >> %t.md5
// RUN: y4=$(%clangxx -x hip --offload-arch=gfx908 -S -emit-llvm -fgpu-rdc %s -o - | md5sum | awk '{ print $1 }') && echo $y4 >> %t.md5
// RUN: y5=$(%clangxx -x hip --offload-arch=gfx908 -S -emit-llvm -fgpu-rdc %s -o - | md5sum | awk '{ print $1 }') && echo $y5 >> %t.md5
// RUN: if grep -qv "$x" %t.md5; then echo "Test failed"; else echo "Test passed"; fi
// CHECK: Test passed
// CHECK-NOT: Test failed

#include "hip/hip_runtime.h"

template<int i>
__attribute__((global)) void kernel() {
  printf("Hello from kernel %d\n", i);
}

template __attribute__((global)) void kernel<1>();
template __attribute__((global)) void kernel<2>();
template __attribute__((global)) void kernel<3>();

int main(int argc, char* argv[]) {
    hipLaunchKernel(reinterpret_cast<void*>(kernel<1>), dim3(1), dim3(1),nullptr, 0, 0);
    hipLaunchKernel(reinterpret_cast<void*>(kernel<2>), dim3(1), dim3(1),nullptr, 0, 0);
    hipLaunchKernel(reinterpret_cast<void*>(kernel<3>), dim3(1), dim3(1),nullptr, 0, 0);
}


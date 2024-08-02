// RUN: x=$(%clang_cc1 -x hip -triple amdgcn -target-cpu gfx908 -emit-llvm -fcuda-is-device %s -o - | md5sum | awk '{ print $1 }') && echo $x
// RUN: y1=$(%clang_cc1 -x hip -triple amdgcn -target-cpu gfx908 -emit-llvm -fcuda-is-device %s -o - | md5sum | awk '{ print $1 }') && echo $y1 >> %t.md5
// RUN: y2=$(%clang_cc1 -x hip -triple amdgcn -target-cpu gfx908 -emit-llvm -fcuda-is-device %s -o - | md5sum | awk '{ print $1 }') && echo $y2 >> %t.md5
// RUN: y3=$(%clang_cc1 -x hip -triple amdgcn -target-cpu gfx908 -emit-llvm -fcuda-is-device %s -o - | md5sum | awk '{ print $1 }') && echo $y3 >> %t.md5
// RUN: y4=$(%clang_cc1 -x hip -triple amdgcn -target-cpu gfx908 -emit-llvm -fcuda-is-device %s -o - | md5sum | awk '{ print $1 }') && echo $y4 >> %t.md5
// RUN: y5=$(%clang_cc1 -x hip -triple amdgcn -target-cpu gfx908 -emit-llvm -fcuda-is-device %s -o - | md5sum | awk '{ print $1 }') && echo $y5 >> %t.md5
// RUN: if grep -qv "$x" %t.md5; then echo "Test failed"; else echo "Test passed"; fi
// CHECK: Test passed
// CHECK-NOT: Test failed

#include "../CodeGenCUDA/Inputs/cuda.h"

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


// RUN: %clang_cc1 -fopenmp -triple amdgcn-amd-amdhsa -fopenmp-is-target-device %s -emit-llvm -S -Wno-openmp-target-exception -o - | FileCheck -check-prefix=DEVICE %s
// RUN: %clang_cc1 -fopenmp -triple x86_64-pc-linux-gnu -fopenmp-is-target-device -fcxx-exceptions %s -emit-llvm -S -Wno-openmp-target-exception -o - | FileCheck -check-prefix=HOST %s
// DEVICE: s_trap
// DEVICE-NOT: __cxa_throw
// HOST: __cxa_throw
// HOST-NOT: s_trap
#pragma omp declare target
void foo(void) {
	throw 404; 
}
#pragma omp end declare target

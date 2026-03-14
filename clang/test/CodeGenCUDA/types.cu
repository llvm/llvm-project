// RUN: %clang_cc1 -triple amdgcn -aux-triple x86_64 -fcuda-is-device -emit-llvm %s -o - | FileCheck -check-prefix=DEV %s
// RUN: %clang_cc1 -triple x86_64 -aux-triple amdgcn -emit-llvm %s -o - | FileCheck -check-prefix=HOST %s

#include "Inputs/cuda.h"

// HOST: @ld_host ={{.*}} global x86_fp80 0xK00000000000000000000
long double ld_host;

// DEV: @ld_device ={{.*}} addrspace(1) externally_initialized global double 0.000000e+00
__device__ long double ld_device;

// Verify that HIP fat binary sections use Mach-O "segment,section" format on Darwin.

// RUN: echo -n "GPU binary would be here." > %t
// RUN: %clang_cc1 -triple x86_64-apple-macosx10.15.0 -emit-llvm %s \
// RUN:     -fcuda-include-gpubinary %t -o - -x hip \
// RUN:   | FileCheck %s --check-prefix=HIPEF
// RUN: %clang_cc1 -cuid=123 -triple x86_64-apple-macosx10.15.0 -emit-llvm %s \
// RUN:     -o - -x hip \
// RUN:   | FileCheck %s --check-prefix=HIPNEF

#include "Inputs/cuda.h"

__device__ int device_var;
__constant__ int constant_var;

// When fat binary is embedded, section names use Mach-O format.
// HIPEF: @[[FATBIN:.*]] = private constant{{.*}} c"GPU binary would be here.",{{.*}}section "__HIP,__hip_fatbin"{{.*}}align 4096
// HIPEF: @__hip_fatbin_wrapper = internal constant { i32, i32, ptr, ptr }
// HIPEF-SAME: section "__HIP,__fatbin"

// When fat binary is external (no -fcuda-include-gpubinary), external symbol uses Mach-O section.
// HIPNEF: @[[FATBIN:__hip_fatbin_[0-9a-f]+]] = external constant i8, section "__HIP,__hip_fatbin"
// HIPNEF: @__hip_fatbin_wrapper = internal constant { i32, i32, ptr, ptr }
// HIPNEF-SAME: section "__HIP,__fatbin"

__global__ void kernelfunc(int i, int j, int k) {}

void hostfunc(void) { kernelfunc<<<1, 1>>>(1, 1, 1); }

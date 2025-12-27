// RUN: clang-tidy %s -checks='-*,modernize-use-nullptr' -- \
// RUN:   --cuda-path=%S/Inputs/CUDA \
// RUN:   -nocudalib -nocudainc -I %S/Inputs/CUDA \
// RUN:   --cuda-host-only | FileCheck %s
// RUN: clang-tidy %s -checks='-*,modernize-use-nullptr' -- \
// RUN:   --cuda-path=%S/Inputs/CUDA \
// RUN:   -nocudalib -nocudainc -I %S/Inputs/CUDA \
// RUN:   --cuda-device-only | FileCheck %s

#include <cuda_runtime.h>

// CHECK: :[[@LINE+1]]:38: warning: use nullptr [modernize-use-nullptr]
__global__ void kernel(int *p) { p = 0; }

// CHECK: :[[@LINE+1]]:11: warning: use nullptr [modernize-use-nullptr]
void *p = 0;

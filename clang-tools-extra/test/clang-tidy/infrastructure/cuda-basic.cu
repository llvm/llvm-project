// RUN: %check_clang_tidy %s modernize-use-nullptr %t -- -- \
// RUN:   --cuda-path=%S/Inputs/CUDA \
// RUN:   -nocudalib -nocudainc -I %S/Inputs/CUDA

#include <cuda.h>

__global__ void kernel(int *p) { p = 0; }
// CHECK-MESSAGES: :[[@LINE-1]]:38: warning: use nullptr [modernize-use-nullptr]
// CHECK-FIXES: __global__ void kernel(int *p) { p = nullptr; }

void *p = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:11: warning: use nullptr [modernize-use-nullptr]
// CHECK-FIXES: void *p = nullptr;

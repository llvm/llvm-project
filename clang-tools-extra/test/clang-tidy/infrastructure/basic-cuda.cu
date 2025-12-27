// RUN: clang-tidy %s -checks='-*,modernize-use-nullptr' -- -nocudainc -nocudalib --cuda-host-only | FileCheck %s

#define __global__ __attribute__((global))

// CHECK: :[[@LINE+1]]:38: warning: use nullptr [modernize-use-nullptr]
__global__ void kernel(int *p) { p = 0; }

// CHECK: :[[@LINE+1]]:11: warning: use nullptr [modernize-use-nullptr]
void *p = 0;

// REQUIRES: hexagon-registered-target
// RUN: %clang -S -target hexagon -fenable-ripple -mhvx -mv79 -fdisable-ripple-lib -emit-llvm %s -o - 2>&1 | FileCheck %s

#include <ripple.h>
#include <ripple_math.h>

#define VEC_SIZE 64
#define N 64

void foo(_Float16 In[N], _Float16 Out[N]) {
  ripple_block_t BS = ripple_set_block_shape(0, VEC_SIZE);
  size_t v = ripple_id(BS, 0);
  Out[v] = tanhf16(In[v]);
}

// CHECK-LABEL: foo
// CHECK: call <64 x half> @llvm.tanh.v64f16
// CHECK: ret void

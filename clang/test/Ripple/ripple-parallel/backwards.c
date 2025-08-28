// REQUIRES: target-x86_64 || target-aarch64 || target=hexagon{{.*}}
// RUN: %clang_cc1 -disable-llvm-passes -emit-llvm -fenable-ripple %s -o - | FileCheck %s

#include <ripple.h>

// 2-dimensional vectorization example
#ifndef HVX_LANE
#define HVX_LANE 0
#endif

#define min(a, b) (a < b ? a : b)

// CHECK: vecadd_subarray
void vecadd_subarray(int N, int start, int end, float x[__restrict 1][N],
                     float y[__restrict 1][N], float xpy[__restrict 1][N]) {
  ripple_block_t BS = ripple_set_block_shape(HVX_LANE, 32, 4);
  // CHECK-COUNT-3: %ripple.par.iv{{[0-9]*}} = alloca i{{[0-9]+}}
  ripple_parallel(BS, 0);
  // CHECK: %[[End:[0-9a-zA-Z_]+]] = load i{{[0-9]+}}, ptr %end.addr
  // CHECK-NEXT: %[[UB:[0-9a-zA-Z_]+]] = sub nsw i{{[0-9]+}} %[[End]], 1
  // CHECK-NEXT: %[[LB:[0-9a-zA-Z_]+]] = load i{{[0-9]+}}, ptr %start.addr
  // CHECK-NEXT: %[[UBMinusLB:[0-9a-zA-Z_]+]] = sub i{{[0-9]+}} %[[UB]], %[[LB]]
  // Because LB is really (LB - 1) because of i >= LB so add 1 to UB - LB
  // CHECK-NEXT: %[[UBMinusLBP1:[0-9a-zA-Z_]+]] = add i{{[0-9]+}} %[[UBMinusLB]], 1
  // CHECK-NEXT: %[[EndMinusStartM1P1D1:[0-9a-zA-Z_]+]] = udiv i{{[0-9]+}} %[[UBMinusLBP1]], 1
  // CHECK-NEXT: store i{{[0-9]+}} %[[EndMinusStartM1P1D1]], ptr %ripple.loop.iters
  for (int i = end - 1; i >= start; --i) {
    int j_max = min(end, 129);
    // CHECK: %[[JMax:[0-9a-zA-Z_]+]] = load i{{[0-9]+}}, ptr %j_max
    // CHECK-NEXT: %[[UB:[0-9a-zA-Z_]+]] = sub nsw i{{[0-9]+}} %[[JMax]], 1
    // CHECK-NEXT: %[[UBMinusLB:[0-9a-zA-Z_]+]] = sub i{{[0-9]+}} %[[UB]], -1
    // CHECK-NEXT: %[[UBMinusLBDivStep:[0-9a-zA-Z_]+]] = udiv i{{[0-9]+}} %[[UBMinusLB]], 1
    // CHECK-NEXT: store i{{[0-9]+}} %[[UBMinusLBDivStep]], ptr %ripple.loop.iters{{[0-9]+}}
    ripple_parallel(BS, 1);
    for (int j = j_max - 1; j >= 0; --j) {
      xpy[i][j] = x[i][j] + y[i][j];
    }
  }
}

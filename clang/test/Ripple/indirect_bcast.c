// REQUIRES: target-x86_64 || target-aarch64 || target=hexagon{{.*}}
// TODO: add appropriate CHECK lines for RUN lines without FileCheck
// RUN: %clang -g -S -fenable-ripple -O0 -emit-llvm %s -o - 2>&1
// RUN: %clang -g -S -fenable-ripple -O1 -emit-llvm %s -o - 2>&1 | FileCheck %s
// RUN: %clang -g -S -fenable-ripple -O2 -emit-llvm %s -o - 2>&1
// RUN: %clang -g -S -fenable-ripple -O3 -emit-llvm %s -o - 2>&1
// RUN: %clang -g -S -fenable-ripple -Os -emit-llvm %s -o - 2>&1
// RUN: %clang -g -S -fenable-ripple -Oz -emit-llvm %s -o - 2>&1

#include <ripple.h>

// CHECK-LABEL: void @indirectStore
// CHECK-SAME: ptr {{.*}} %[[IN:[a-zA-Z0-9]*]], ptr {{.*}} %[[OUT:[a-zA-Z0-9]*]]
void indirectStore(float In[32] , float (*Out[32])[32]) {
  ripple_block_t BS = ripple_set_block_shape(0, 32);
  size_t BlockIdx = ripple_id(BS, 0);
  for (unsigned i = 0; i < 32; ++i) {
    // CHECK: %[[IndVar:[a-zA-Z0-9_.]+]] = phi i{{[0-9]+}} [ 0, %{{.*}} ], [ %{{[a-zA-Z0-9_.]+}}, %{{.*}} ]
    // CHECK: %[[InLoad:[a-zA-Z0-9_.]+]] = load <32 x float>, ptr %[[IN]]
    // CHECK: %[[OutOffset:[a-zA-Z0-9_.]+]] = getelementptr inbounds nuw ptr, ptr %[[OUT]], i{{[0-9]+}} %[[IndVar]]
    // CHECK: %[[OutAddr:[a-zA-Z0-9_.]+]] = load ptr, ptr %[[OutOffset]]
    // CHECK: store <32 x float> %[[InLoad]], ptr %[[OutAddr]]
    (*Out[i])[BlockIdx] = In[BlockIdx];
  }
}

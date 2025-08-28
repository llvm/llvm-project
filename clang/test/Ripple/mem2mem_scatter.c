// REQUIRES: target-x86_64 || target-aarch64 || target=hexagon{{.*}}
// TODO: add appropriate CHECK lines for RUN lines without FileCheck
// RUN: %clang -g -S -fenable-ripple -O0 -emit-llvm %s -o %t
// RUN: %clang -g -S -fenable-ripple -O1 -emit-llvm %s -o %t
// RUN: %clang -g -S -fenable-ripple -O2 -emit-llvm %s -o - 2>&1 | FileCheck %s
// RUN: %clang -g -S -fenable-ripple -Os -emit-llvm %s -o - 2>&1 | FileCheck %s
// RUN: %clang -g -S -fenable-ripple -Oz -emit-llvm %s -o %t
// RUN: %clang -g -S -fenable-ripple -O3 -emit-llvm %s -o %t

#include <ripple.h>

// CHECK-LABEL: void @memory_to_memory_scatter
// CHECK-SAME: ptr {{.*}} [[IN:%.*]], ptr {{.*}} [[OUT:%.*]])
void memory_to_memory_scatter(float *In[32], float *Out[32][32]) {
  // CHECK: [[GatherAddrs:%.*]] = load <32 x ptr>, ptr [[IN]]
  // CHECK: [[InGather:%.*]] = tail call <32 x float> @llvm.masked.gather.v32f32.v32p0(<32 x ptr> align 4 [[GatherAddrs]], <32 x i1> splat (i1 true), <32 x float> poison)
  ripple_block_t BS = ripple_set_block_shape(0, 32);
  size_t BlockIdx = ripple_id(BS, 0);
  #pragma clang loop unroll(disable)
  for (unsigned i = 0; i < 32; ++i) {
    // CHECK: [[IndVar:%.*]] = phi i{{[0-9]+}} [ 0, {{.*}} ], [ %{{.*}}, %{{.*}} ]
    // CHECK: [[GEP:%.*]] = getelementptr{{.*}}[32 x ptr], ptr %Out, i{{[0-9]+}} [[IndVar]]
    // CHECK: [[ScatterAddrs:%.*]] = load <32 x ptr>, ptr [[GEP]]
    // CHECK: tail call void @llvm.masked.scatter.v32f32.v32p0(<32 x float> [[InGather]], <32 x ptr> align 4 [[ScatterAddrs]], <32 x i1> splat (i1 true))
    *(Out[i][BlockIdx]) = *(In[BlockIdx]);
  }
}

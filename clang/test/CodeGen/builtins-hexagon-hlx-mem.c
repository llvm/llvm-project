// RUN: %clang_cc1 -triple hexagon -target-cpu hexagonv85 -target-feature +hlx -emit-llvm -o - %s | FileCheck %s

// Test HLX memory intrinsics.

typedef short HLX_Vector __attribute__((__vector_size__(2048))) __attribute__((aligned(2048)));
typedef long HLX_VectorPair __attribute__((__vector_size__(4096))) __attribute__((aligned(2048)));

// CHECK-LABEL: @test0
// CHECK: call <1024 x i16> @llvm.hexagon.LX.L16b.tmp(ptr %0)
HLX_Vector test0(void *BaseAddress) {
  return __builtin_HEXAGON_LX_L16b_tmp(BaseAddress);
}

// CHECK-LABEL: @test1
// CHECK: call void @llvm.hexagon.LX.S16b.new(ptr %0, <1024 x i16> %1)
void test1(void *BaseAddress, HLX_Vector V) {
  return __builtin_HEXAGON_LX_S16b_new(BaseAddress, V);
}

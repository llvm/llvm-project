// RUN: %clang_cc1 -triple hexagon -target-cpu hexagonv85 -target-feature +hlx -emit-llvm -o - %s | FileCheck %s

// Test HLX intrinsics.

typedef short HLX_Vector __attribute__((__vector_size__(2048))) __attribute__((aligned(2048)));
typedef long HLX_VectorPair __attribute__((__vector_size__(4096))) __attribute__((aligned(2048)));

// CHECK-LABEL: @test0
// CHECK: call <1024 x i16> @llvm.hexagon.LX.hi(<1024 x i32>
HLX_Vector test0(HLX_VectorPair a) {
  return __builtin_HEXAGON_LX_hi(a);
}

// CHECK-LABEL: @test1
// CHECK: call <1024 x i16> @llvm.hexagon.LX.lo(<1024 x i32>
HLX_Vector test1(HLX_VectorPair a) {
  return __builtin_HEXAGON_LX_lo(a);
}

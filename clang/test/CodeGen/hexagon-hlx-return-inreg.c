// REQUIRES: hexagon-registered-target
// RUN: %clang_cc1 -triple hexagon -target-cpu hexagonv85 -target-feature +hlx \
// RUN:   -emit-llvm %s -o - | FileCheck %s

// HLX vectors and vector pairs are returned in HLX registers, not via memory.

typedef short HLX_Vector __attribute__((__vector_size__(2048))) __attribute__((aligned(2048)));
typedef long HLX_VectorPair __attribute__((__vector_size__(4096))) __attribute__((aligned(2048)));

// CHECK: define dso_local inreg <1024 x i16> @hlx_add
// CHECK: ret <1024 x i16>
HLX_Vector hlx_add(HLX_Vector a, HLX_Vector b) {
  return a + b;
}

// CHECK: define dso_local inreg <1024 x i32> @hlx_pair_add
// CHECK: ret <1024 x i32>
HLX_VectorPair hlx_pair_add(HLX_VectorPair a, HLX_VectorPair b) {
  return a + b;
}

// REQUIRES: target=hexagon{{.*}} || target-x86_64
// RUN: %clang -S -O2 -emit-llvm %s -DCOMPILE_LIB=1 -o %t.ll
// RUN: %clang -S -O2 -emit-llvm -fenable-ripple -fripple-lib=%t.ll -o - %s | FileCheck %s

#ifdef COMPILE_LIB

typedef float v4f32 __attribute__((__vector_size__(16)))
__attribute__((aligned(16)));
typedef float v16f32 __attribute__((__vector_size__(64)))
__attribute__((aligned(16)));

// Valid
extern v4f32 ripple_uniform_t1x4f32_externf1(v4f32 A, v4f32 B, float C) { return (A / B) + C; }
extern v4f32 ripple_uniform_t1x4f32_externf2(v4f32 A, float C, v4f32 B) { return (A / B) + C; }
extern v4f32 ripple_ret_t1x4f32_t1x4f32_t4f32_externf3(v4f32 A, v4f32 B) { return A / B; }
extern v16f32 ripple_ret_t4x4f32_t1x4f32_t4f32_externf4(v4f32 A, v4f32 B) { v16f32 v = {8.f, 1.f}; return v; }
extern v16f32 ripple_pure_ew_uniform_t4x4f32_externf5(v16f32 A, v16f32 B) { return A + B; }
extern v16f32 ripple_ew_uniform_t4x4f32_externf5_autopure(v16f32 A, v16f32 B) { return A + B; }

#else

extern float externf1(float, float, float);
extern float externf2(float, float, float);
extern float externf3(float, float);
extern float externf4(float, float);
extern float externf5(float, float);
extern float externf5_autopure(float, float);

#include <ripple.h>

// CHECK-LABEL: define dso_local void @test_valid_candidates
// CHECK: call{{.*}}@ripple_uniform_t1x4f32_externf1
// CHECK: call{{.*}}@ripple_uniform_t1x4f32_externf2
// CHECK: call{{.*}}@ripple_ret_t1x4f32_t1x4f32_t4f32_externf3
// CHECK: call{{.*}}@ripple_ret_t4x4f32_t1x4f32_t4f32_externf4
// CHECK: call{{.*}}@ripple_pure_ew_uniform_t4x4f32_externf5
// CHECK: call{{.*}}@ripple_ew_uniform_t4x4f32_externf5_autopure
// CHECK: call{{.*}}@ripple_pure_ew_uniform_t4x4f32_externf5
// CHECK: call{{.*}}@ripple_pure_ew_uniform_t4x4f32_externf5
// CHECK: call{{.*}}@ripple_pure_ew_uniform_t4x4f32_externf5
// CHECK: call{{.*}}@ripple_pure_ew_uniform_t4x4f32_externf5
// CHECK: call{{.*}}@ripple_ew_uniform_t4x4f32_externf5_autopure
void test_valid_candidates(float *restrict f, float *restrict f2, float *restrict f3) {
  ripple_block_t BS = ripple_set_block_shape(0, 4, 4);
  size_t RID = ripple_id(BS, 0);
  size_t RID2 = ripple_id(BS, 1);
  ripple_block_t BS2 = ripple_set_block_shape(0, 16, 16, 16);

  f[RID2] = externf1(f2[RID2], f3[RID2], 42.f);
  f[RID2] += externf2(f2[RID2], f[0], f3[RID2]);
  f[RID2] += externf3(f2[RID2], f3[RID]);
  size_t TwodRID = RID + ripple_get_block_size(BS, 0) * RID2;
  f[TwodRID] += externf4(f2[RID2], f3[RID]);

  // Testing elementwise
  f[TwodRID] += externf5(f2[TwodRID], f3[TwodRID]);
  f[TwodRID] += externf5_autopure(f2[TwodRID], f3[TwodRID]);
  // EW accepts any shape, assuming the tensor element count match and the shapes are the same
  f[ripple_id(BS2, 0)] += externf5(f2[ripple_id(BS2, 0)], f3[ripple_id(BS2, 0)]);
  f[ripple_id(BS2, 1)] += externf5(f2[ripple_id(BS2, 1)], f3[ripple_id(BS2, 1)]);
  f[ripple_id(BS2, 2)] += externf5(f2[ripple_id(BS2, 2)], f3[ripple_id(BS2, 2)]);

  if (TwodRID < 10) {
    f[TwodRID] += externf5(f2[TwodRID], f3[TwodRID]);
    f[TwodRID] += externf5_autopure(f2[TwodRID], f3[TwodRID]);
  }
}

// CHECK-LABEL: define dso_local void @test_no_candidate
// CHECK: call float @externf3
void test_no_candidate(float *f, float *f2, float *f3) {
  ripple_block_t BS = ripple_set_block_shape(0, 4);
  size_t RID = ripple_id(BS, 0);

  f[RID] += externf3(f2[RID], f3[RID]);
}

#endif // COMPILE_LIB

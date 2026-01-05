// REQUIRES: aarch64-registered-target
// RUN: %clang -ffreestanding --target=aarch64-unknown-linux-gnu -S -O2 -emit-llvm %s -DCOMPILE_LIB=1 -o %t.ll
// RUN: %clang -ffreestanding --target=aarch64-unknown-linux-gnu -S -O2 -emit-llvm -fenable-ripple -fripple-lib=%t.ll -mllvm -ripple-disable-link -o - %s | FileCheck %s

#ifdef COMPILE_LIB

typedef float v4f32 __attribute__((__vector_size__(16)))
__attribute__((aligned(16)));
typedef float v16f32 __attribute__((__vector_size__(64)))
__attribute__((aligned(16)));

extern v4f32 ripple_externf1(void) { v4f32 v = {10.f, 12.f}; return v; }
extern v4f32 ripple_uniform_t1x4f32_externf2(void) { v4f32 v = {10.f, 12.f}; return v; }
extern v4f32 ripple_ret_t1x4f32_externf3(void) { v4f32 v = {10.f, 12.f}; return v; }
extern v16f32 ripple_ret_t4x4f32_externf4(void) { v16f32 v = {8.f, 1.f}; return v; }
extern v16f32 ripple_pure_uniform_t4x4f32_externf5(void) { v16f32 v = {8.f, 1.f}; return v; }

#else

#include <ripple.h>

extern float externf1(ripple_block_t BS);
extern float externf2(ripple_block_t BS);
extern float externf3(ripple_block_t BS);
extern float externf4(ripple_block_t BS);
extern float externf5(ripple_block_t BS);

// CHECK-LABEL: define dso_local void @test_valid_candidates(
// CHECK-NEXT:  [[ENTRY:.*:]]
// CHECK:    [[TMP6:%.*]] = tail call <4 x float> @ripple_externf1()
// CHECK:    [[TMP7:%.*]] = tail call <4 x float> @ripple_uniform_t1x4f32_externf2()
// CHECK:    [[TMP8:%.*]] = tail call <4 x float> @ripple_ret_t1x4f32_externf3()
// CHECK: call void @ripple_ret_t4x4f32_externf4(ptr nonnull
// CHECK-COUNT-5: call void @ripple_pure_uniform_t4x4f32_externf5(ptr nonnull
//
void test_valid_candidates(float *restrict f, float *restrict f2, float *restrict f3) {
  ripple_block_t BS = ripple_set_block_shape(0, 4, 4);
  size_t RID = ripple_id(BS, 0);
  size_t RID2 = ripple_id(BS, 1);
  ripple_block_t BS2 = ripple_set_block_shape(0, 16, 16, 16);

  // f[RID2] = externf1(BS);
  // f[RID2] += externf2(BS);
  // f[RID2] /= externf3(BS);

  f[RID] = externf1(BS);
  f[RID2] += externf2(BS);
  f[RID2] /= externf3(BS);
  size_t TwodRID = RID + ripple_get_block_size(BS, 0) * RID2;
  f[TwodRID] += externf4(BS);

  // Testing elementwise
  f[TwodRID] += externf5(BS);
  // EW accepts any shape, assuming the tensor element count match and the shapes are the same
  f[ripple_id(BS, 0) + 4 * ripple_id(BS, 1)] += externf5(BS);
  f[ripple_id(BS, 0) + 4 * ripple_id(BS, 1)] += externf5(BS);
  f[ripple_id(BS, 0) + 4 * ripple_id(BS, 1)] += externf5(BS);

  if (TwodRID < 10)
    f[TwodRID] += externf5(BS);
}

#endif // COMPILE_LIB

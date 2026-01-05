// REQUIRES: aarch64-registered-target
// RUN: %clang -ffreestanding --target=aarch64-unknown-linux-gnu -S -O2 -emit-llvm %s -DCOMPILE_LIB=1 -o %t.ll
// RUN: %clang -ffreestanding --target=aarch64-unknown-linux-gnu -S -O2 -emit-llvm -fenable-ripple -fripple-lib=%t.ll -mllvm -ripple-disable-link -o - %s | FileCheck %s

#ifdef COMPILE_LIB

typedef float v4f32 __attribute__((__vector_size__(16)))
__attribute__((aligned(16)));
typedef float v16f32 __attribute__((__vector_size__(64)))
__attribute__((aligned(16)));

extern v4f32 ripple_scalar_tensor1(int x) {
  v4f32 v = {(float)x, (float)x * 2.f};
  return v;
}

extern v4f32 ripple_uniform_t1x4f32_scalar_tensor2(float a, int b) {
  v4f32 v = {a, (float)b};
  return v;
}

extern v4f32 ripple_ret_t1x4f32_scalar_tensor3(float val) {
  v4f32 v = {val, val * 2.f};
  return v;
}

extern v16f32 ripple_ret_t4x4f32_scalar_tensor4(int a, float b, int c) {
  v16f32 v = {(float)a, b, (float)c};
  return v;
}

extern v16f32 ripple_pure_uniform_t4x4f32_scalar_tensor5(double d) {
  v16f32 v = {(float)d, (float)d * 0.5f};
  return v;
}

#else

#include <ripple.h>

extern float scalar_tensor1(ripple_block_t BS, int x);
extern float scalar_tensor2(ripple_block_t BS, float a, int b);
extern float scalar_tensor3(ripple_block_t BS, float val);
extern float scalar_tensor4(ripple_block_t BS, int a, float b, int c);
extern float scalar_tensor5(ripple_block_t BS, double d);

// CHECK-LABEL: define dso_local void @test_scalar_tensor_candidates(
// CHECK-NEXT:  [[ENTRY:.*:]]
// CHECK:    [[TMP6:%.*]] = tail call <4 x float> @ripple_scalar_tensor1(i32 42)
// CHECK:    [[TMP7:%.*]] = tail call <4 x float> @ripple_uniform_t1x4f32_scalar_tensor2(float {{.*}}, i32 10)
// CHECK:    [[TMP8:%.*]] = tail call <4 x float> @ripple_ret_t1x4f32_scalar_tensor3(float {{.*}})
// CHECK: call void @ripple_ret_t4x4f32_scalar_tensor4(ptr nonnull {{.*}}, i32 1, float {{.*}}, i32 3)
// CHECK-COUNT-5: call void @ripple_pure_uniform_t4x4f32_scalar_tensor5(ptr nonnull {{.*}}, double {{.*}})
//
void test_scalar_tensor_candidates(float *restrict f, float *restrict f2, float *restrict f3) {
  ripple_block_t BS = ripple_set_block_shape(0, 4, 4);
  size_t RID = ripple_id(BS, 0);
  size_t RID2 = ripple_id(BS, 1);

  // Test with single scalar argument
  f[RID] = scalar_tensor1(BS, 42);

  // Test with multiple scalar arguments
  f[RID2] += scalar_tensor2(BS, 3.14f, 10);

  // Test with float scalar
  f[RID2] /= scalar_tensor3(BS, 2.5f);

  size_t TwodRID = RID + ripple_get_block_size(BS, 0) * RID2;

  // Test with three scalar arguments returning larger tensor
  f[TwodRID] += scalar_tensor4(BS, 1, 1.5f, 3);

  // Testing elementwise with scalar argument
  f[TwodRID] += scalar_tensor5(BS, 6.28);
  // EW accepts any shape, assuming the tensor element count match and the shapes are the same
  f[ripple_id(BS, 0) + 4 * ripple_id(BS, 1)] += scalar_tensor5(BS, 3.14);
  f[ripple_id(BS, 0) + 4 * ripple_id(BS, 1)] += scalar_tensor5(BS, 2.71);
  f[ripple_id(BS, 0) + 4 * ripple_id(BS, 1)] += scalar_tensor5(BS, 1.41);

  if (TwodRID < 10)
    f[TwodRID] += scalar_tensor5(BS, 9.81);
}

#endif // COMPILE_LIB

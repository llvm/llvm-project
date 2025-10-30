// REQUIRES: aarch64-registered-target
// RUN: %clang --target=aarch64-unknown-linux-gnu -S -O2 -emit-llvm %s -DCOMPILE_LIB=1 -o %t.ll
// RUN: %clang --target=aarch64-unknown-linux-gnu -g -S -O2 -emit-llvm -fenable-ripple -fripple-lib=%t.ll -mllvm -ripple-disable-link %s -ferror-limit=0 2>%t.err; FileCheck %s --input-file=%t.err

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

void test1(float *restrict f, float *restrict f2, float *restrict f3) {
  ripple_block_t WrongBS = ripple_set_block_shape(0, 8, 2);
  ripple_block_t BS = ripple_set_block_shape(0, 4, 4);
  size_t RID = ripple_id(BS, 0);
  size_t RID2 = ripple_id(BS, 1);
  ripple_block_t BS2 = ripple_set_block_shape(0, 16, 16, 16);

  f[RID] = externf1(WrongBS);
}

void test2(float *restrict f, float *restrict f2, float *restrict f3) {
  ripple_block_t WrongBS = ripple_set_block_shape(0, 8, 2);
  ripple_block_t BS = ripple_set_block_shape(0, 4, 4);
  size_t RID = ripple_id(BS, 0);
  size_t RID2 = ripple_id(BS, 1);
  ripple_block_t BS2 = ripple_set_block_shape(0, 16, 16, 16);

  f[RID2] += externf2(WrongBS);
  f[RID2] /= externf3(WrongBS);
  size_t TwodRID = RID + ripple_get_block_size(BS, 0) * RID2;
  f[TwodRID] += externf4(WrongBS);

  // Testing elementwise
  f[TwodRID] += externf5(WrongBS);
  // EW accepts any shape, assuming the tensor element count match and the shapes are the same
  f[ripple_id(BS, 0) + 4 * ripple_id(BS, 1)] += externf5(WrongBS);

  if (TwodRID < 10)
    f[TwodRID] += externf5(WrongBS);
}

void test3(float *restrict f, float *restrict f2, float *restrict f3) {
  ripple_block_t WrongBS = ripple_set_block_shape(0, 8, 2);
  ripple_block_t BS = ripple_set_block_shape(0, 4, 4);
  size_t RID = ripple_id(BS, 0);
  size_t RID2 = ripple_id(BS, 1);
  ripple_block_t BS2 = ripple_set_block_shape(0, 16, 16, 16);

  f[RID2] /= externf3(WrongBS);
  size_t TwodRID = RID + ripple_get_block_size(BS, 0) * RID2;
  f[TwodRID] += externf4(WrongBS);

  // Testing elementwise
  f[TwodRID] += externf5(WrongBS);
  // EW accepts any shape, assuming the tensor element count match and the shapes are the same
  f[ripple_id(BS, 0) + 4 * ripple_id(BS, 1)] += externf5(WrongBS);

  if (TwodRID < 10)
    f[TwodRID] += externf5(WrongBS);
}

void test4(float *restrict f, float *restrict f2, float *restrict f3) {
  ripple_block_t WrongBS = ripple_set_block_shape(0, 8, 2);
  ripple_block_t BS = ripple_set_block_shape(0, 4, 4);
  size_t RID = ripple_id(BS, 0);
  size_t RID2 = ripple_id(BS, 1);
  ripple_block_t BS2 = ripple_set_block_shape(0, 16, 16, 16);

  size_t TwodRID = RID + ripple_get_block_size(BS, 0) * RID2;
  f[TwodRID] += externf4(WrongBS);

  // Testing elementwise
  f[TwodRID] += externf5(WrongBS);
  // EW accepts any shape, assuming the tensor element count match and the shapes are the same
  f[ripple_id(BS, 0) + 4 * ripple_id(BS, 1)] += externf5(WrongBS);

  if (TwodRID < 10)
    f[TwodRID] += externf5(WrongBS);
}

void test5(float *restrict f, float *restrict f2, float *restrict f3) {
  ripple_block_t WrongBS = ripple_set_block_shape(0, 8, 2);
  ripple_block_t BS = ripple_set_block_shape(0, 4, 4);
  size_t RID = ripple_id(BS, 0);
  size_t RID2 = ripple_id(BS, 1);
  ripple_block_t BS2 = ripple_set_block_shape(0, 16, 16, 16);

  size_t TwodRID = RID + ripple_get_block_size(BS, 0) * RID2;

  // Testing elementwise
  f[TwodRID] += externf5(WrongBS);

  if (TwodRID < 10)
    f[TwodRID] += externf5(WrongBS);
}

void test6(float *restrict f, float *restrict f2, float *restrict f3) {
  ripple_block_t WrongBS = ripple_set_block_shape(0, 8, 2);
  ripple_block_t BS = ripple_set_block_shape(0, 4, 4);
  size_t RID = ripple_id(BS, 0);
  size_t RID2 = ripple_id(BS, 1);
  ripple_block_t BS2 = ripple_set_block_shape(0, 16, 16, 16);

  size_t TwodRID = RID + ripple_get_block_size(BS, 0) * RID2;

  if (TwodRID < 10)
    f[TwodRID] += externf5(WrongBS);
}

// CHECK:      external-void-selection-wrong-block-shape.c:124:19: error: no matching external ripple function found; ensure the block shape is compatible with one of the declared external functions
// CHECK-NEXT:   124 |     f[TwodRID] += externf5(WrongBS);
// CHECK:      external-void-selection-wrong-block-shape.c:115:28: note: using this block with shape Tensor[8][2]
// CHECK-NEXT:   115 |   ripple_block_t WrongBS = ripple_set_block_shape(0, 8, 2);

// CHECK:      external-void-selection-wrong-block-shape.c:108:17: error: no matching external ripple function found; ensure the block shape is compatible with one of the declared external functions
// CHECK-NEXT:   108 |   f[TwodRID] += externf5(WrongBS);
// CHECK:      external-void-selection-wrong-block-shape.c:99:28: note: using this block with shape Tensor[8][2]
// CHECK-NEXT:    99 |   ripple_block_t WrongBS = ripple_set_block_shape(0, 8, 2);
// CHECK:      external-void-selection-wrong-block-shape.c:111:19: error: no matching external ripple function found; ensure the block shape is compatible with one of the declared external functions
// CHECK-NEXT:   111 |     f[TwodRID] += externf5(WrongBS);
// CHECK:      external-void-selection-wrong-block-shape.c:99:28: note: using this block with shape Tensor[8][2]
// CHECK-NEXT:    99 |   ripple_block_t WrongBS = ripple_set_block_shape(0, 8, 2);

// CHECK:      external-void-selection-wrong-block-shape.c:87:17: error: no matching external ripple function found; ensure the block shape is compatible with one of the declared external functions
// CHECK-NEXT:    87 |   f[TwodRID] += externf4(WrongBS);
// CHECK:      external-void-selection-wrong-block-shape.c:80:28: note: using this block with shape Tensor[8][2]
// CHECK-NEXT:    80 |   ripple_block_t WrongBS = ripple_set_block_shape(0, 8, 2);
// CHECK:      external-void-selection-wrong-block-shape.c:90:17: error: no matching external ripple function found; ensure the block shape is compatible with one of the declared external functions
// CHECK-NEXT:    90 |   f[TwodRID] += externf5(WrongBS);
// CHECK:      external-void-selection-wrong-block-shape.c:80:28: note: using this block with shape Tensor[8][2]
// CHECK-NEXT:    80 |   ripple_block_t WrongBS = ripple_set_block_shape(0, 8, 2);
// CHECK:      external-void-selection-wrong-block-shape.c:92:49: error: no matching external ripple function found; ensure the block shape is compatible with one of the declared external functions
// CHECK-NEXT:    92 |   f[ripple_id(BS, 0) + 4 * ripple_id(BS, 1)] += externf5(WrongBS);
// CHECK:      external-void-selection-wrong-block-shape.c:80:28: note: using this block with shape Tensor[8][2]
// CHECK-NEXT:    80 |   ripple_block_t WrongBS = ripple_set_block_shape(0, 8, 2);
// CHECK:      external-void-selection-wrong-block-shape.c:95:19: error: no matching external ripple function found; ensure the block shape is compatible with one of the declared external functions
// CHECK-NEXT:    95 |     f[TwodRID] += externf5(WrongBS);
// CHECK:      external-void-selection-wrong-block-shape.c:80:28: note: using this block with shape Tensor[8][2]
// CHECK-NEXT:    80 |   ripple_block_t WrongBS = ripple_set_block_shape(0, 8, 2);

// CHECK:      external-void-selection-wrong-block-shape.c:66:14: error: no matching external ripple function found; ensure the block shape is compatible with one of the declared external functions
// CHECK-NEXT:    66 |   f[RID2] /= externf3(WrongBS);
// CHECK:      external-void-selection-wrong-block-shape.c:60:28: note: using this block with shape Tensor[8][2]
// CHECK-NEXT:    60 |   ripple_block_t WrongBS = ripple_set_block_shape(0, 8, 2);
// CHECK:      external-void-selection-wrong-block-shape.c:68:17: error: no matching external ripple function found; ensure the block shape is compatible with one of the declared external functions
// CHECK-NEXT:    68 |   f[TwodRID] += externf4(WrongBS);
// CHECK:      external-void-selection-wrong-block-shape.c:60:28: note: using this block with shape Tensor[8][2]
// CHECK-NEXT:    60 |   ripple_block_t WrongBS = ripple_set_block_shape(0, 8, 2);
// CHECK:      external-void-selection-wrong-block-shape.c:71:17: error: no matching external ripple function found; ensure the block shape is compatible with one of the declared external functions
// CHECK-NEXT:    71 |   f[TwodRID] += externf5(WrongBS);
// CHECK:      external-void-selection-wrong-block-shape.c:60:28: note: using this block with shape Tensor[8][2]
// CHECK-NEXT:    60 |   ripple_block_t WrongBS = ripple_set_block_shape(0, 8, 2);
// CHECK:      external-void-selection-wrong-block-shape.c:73:49: error: no matching external ripple function found; ensure the block shape is compatible with one of the declared external functions
// CHECK-NEXT:    73 |   f[ripple_id(BS, 0) + 4 * ripple_id(BS, 1)] += externf5(WrongBS);
// CHECK:      external-void-selection-wrong-block-shape.c:60:28: note: using this block with shape Tensor[8][2]
// CHECK-NEXT:    60 |   ripple_block_t WrongBS = ripple_set_block_shape(0, 8, 2);

// CHECK:      external-void-selection-wrong-block-shape.c:45:14: error: no matching external ripple function found; ensure the block shape is compatible with one of the declared external functions
// CHECK-NEXT:    45 |   f[RID2] += externf2(WrongBS);
// CHECK:      external-void-selection-wrong-block-shape.c:39:28: note: using this block with shape Tensor[8][2]
// CHECK-NEXT:    39 |   ripple_block_t WrongBS = ripple_set_block_shape(0, 8, 2);
// CHECK:      external-void-selection-wrong-block-shape.c:46:14: error: no matching external ripple function found; ensure the block shape is compatible with one of the declared external functions
// CHECK-NEXT:    46 |   f[RID2] /= externf3(WrongBS);
// CHECK:      external-void-selection-wrong-block-shape.c:39:28: note: using this block with shape Tensor[8][2]
// CHECK-NEXT:    39 |   ripple_block_t WrongBS = ripple_set_block_shape(0, 8, 2);
// CHECK:      external-void-selection-wrong-block-shape.c:48:17: error: no matching external ripple function found; ensure the block shape is compatible with one of the declared external functions
// CHECK-NEXT:    48 |   f[TwodRID] += externf4(WrongBS);
// CHECK:      external-void-selection-wrong-block-shape.c:39:28: note: using this block with shape Tensor[8][2]
// CHECK-NEXT:    39 |   ripple_block_t WrongBS = ripple_set_block_shape(0, 8, 2);
// CHECK:      external-void-selection-wrong-block-shape.c:51:17: error: no matching external ripple function found; ensure the block shape is compatible with one of the declared external functions
// CHECK-NEXT:    51 |   f[TwodRID] += externf5(WrongBS);
// CHECK:      external-void-selection-wrong-block-shape.c:39:28: note: using this block with shape Tensor[8][2]
// CHECK-NEXT:    39 |   ripple_block_t WrongBS = ripple_set_block_shape(0, 8, 2);
// CHECK:      external-void-selection-wrong-block-shape.c:53:49: error: no matching external ripple function found; ensure the block shape is compatible with one of the declared external functions
// CHECK-NEXT:    53 |   f[ripple_id(BS, 0) + 4 * ripple_id(BS, 1)] += externf5(WrongBS);
// CHECK:      external-void-selection-wrong-block-shape.c:39:28: note: using this block with shape Tensor[8][2]
// CHECK-NEXT:    39 |   ripple_block_t WrongBS = ripple_set_block_shape(0, 8, 2);
// CHECK:      external-void-selection-wrong-block-shape.c:56:19: error: no matching external ripple function found; ensure the block shape is compatible with one of the declared external functions
// CHECK-NEXT:    56 |     f[TwodRID] += externf5(WrongBS);
// CHECK:      external-void-selection-wrong-block-shape.c:39:28: note: using this block with shape Tensor[8][2]
// CHECK-NEXT:    39 |   ripple_block_t WrongBS = ripple_set_block_shape(0, 8, 2);

// CHECK:      external-void-selection-wrong-block-shape.c:35:12: error: no matching external ripple function found; ensure the block shape is compatible with one of the declared external functions
// CHECK-NEXT:    35 |   f[RID] = externf1(WrongBS);
// CHECK:      external-void-selection-wrong-block-shape.c:29:28: note: using this block with shape Tensor[8][2]
// CHECK-NEXT:    29 |   ripple_block_t WrongBS = ripple_set_block_shape(0, 8, 2);

#endif // COMPILE_LIB

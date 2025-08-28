// NOTE:
// REQUIRES: target-x86_64 || target-aarch64 || target=hexagon{{.*}}
// RUN: %clang -Xclang -fexperimental-strict-floating-point -Wall -Wextra -Wpedantic -Wripple -fenable-ripple -O2 -S -emit-llvm %s -o - 2>&1 | FileCheck %s --implicit-check-not="warning:"

#include <ripple.h>

// CHECK-LABEL: define dso_local void @test1(
// CHECK-SAME: ptr{{.*}}[[MAX:%.*]], ptr{{.*}}[[MIN:%.*]], ptr{{.*}}[[REALMAX:%.*]], ptr{{.*}}[[REALMIN:%.*]]){{.*}}{
// CHECK-NEXT:  [[ENTRY:.*:]]
// CHECK-NEXT:    [[DOTRIPPLE_REDUCTION:%.*]] = tail call reassoc float @llvm.vector.reduce.fmax.v3f32(<3 x float> <float 0x40091EB860000000, float 0x40191EB860000000, float 0x4022D70A40000000>)
// CHECK-NEXT:    store float [[DOTRIPPLE_REDUCTION]], ptr [[MAX]], align 4, !tbaa [[TBAA3:![0-9]+]]
// CHECK-NEXT:    [[DOTRIPPLE_REDUCTION5:%.*]] = tail call reassoc float @llvm.vector.reduce.fmin.v3f32(<3 x float> <float 0x40091EB860000000, float 0x40191EB860000000, float 0x4022D70A40000000>)
// CHECK-NEXT:    store float [[DOTRIPPLE_REDUCTION5]], ptr [[MIN]], align 4, !tbaa [[TBAA3]]
// CHECK-NEXT:    store float 0x4022D70A40000000, ptr [[REALMAX]], align 4, !tbaa [[TBAA3]]
// CHECK-NEXT:    store float 0x40091EB860000000, ptr [[REALMIN]], align 4, !tbaa [[TBAA3]]
// CHECK-NEXT:    ret void
//
void test1(float *max, float *min, float *realMax, float *realMin) {
  float array[3] = {3.14, 3.14 * 2, 3.14 * 3};

  ripple_block_t BS = ripple_set_block_shape(0, 3);
  size_t v = ripple_id(BS, 0);
  float x = array[v];

  *max = ripple_reducemax(0b1, x);
  *min = ripple_reducemin(0b1, x);
  *realMax = array[2];
  *realMin = array[0];
}


#pragma float_control(push)
#pragma float_control(precise, off)

extern float sqrtf(float);

// CHECK-LABEL: define dso_local range(i32 0, 2) i32 @testNan(
// CHECK-SAME: ptr{{.*}}[[MAX:%.*]], ptr{{.*}}[[MIN:%.*]], ptr{{.*}}[[REALMAX:%.*]], ptr{{.*}}[[REALMIN:%.*]]){{.*}}{
// CHECK-NEXT:  [[ENTRY:.*:]]
// CHECK-NEXT:    [[DOTRIPPLE_REDUCTION:%.*]] = tail call fast float @llvm.vector.reduce.fmax.v3f32(<3 x float> <float 0x7FF8000000000000, float 0x40191EB860000000, float 0x4022D70A40000000>)
// CHECK-NEXT:    store float [[DOTRIPPLE_REDUCTION]], ptr [[MAX]], align 4, !tbaa [[TBAA3]]
// CHECK-NEXT:    [[DOTRIPPLE_REDUCTION12:%.*]] = tail call fast float @llvm.vector.reduce.fmin.v3f32(<3 x float> <float 0x7FF8000000000000, float 0x40191EB860000000, float 0x4022D70A40000000>)
// CHECK-NEXT:    store float [[DOTRIPPLE_REDUCTION12]], ptr [[MIN]], align 4, !tbaa [[TBAA3]]
// CHECK-NEXT:    store float 0x4022D70A40000000, ptr [[REALMAX]], align 4, !tbaa [[TBAA3]]
// CHECK-NEXT:    store float 0x40191EB860000000, ptr [[REALMIN]], align 4, !tbaa [[TBAA3]]
// CHECK-NEXT:    [[TMP0:%.*]] = load float, ptr [[MAX]], align 4, !tbaa [[TBAA3]]
// CHECK-NEXT:    [[TMP1:%.*]] = load float, ptr [[REALMAX]], align 4, !tbaa [[TBAA3]]
// CHECK-NEXT:    [[CMP:%.*]] = fcmp fast oeq float [[TMP0]], [[TMP1]]
// CHECK-NEXT:    [[TMP2:%.*]] = load float, ptr [[MIN]], align 4, !tbaa [[TBAA3]]
// CHECK-NEXT:    [[CMP3:%.*]] = fcmp fast oeq float [[TMP2]], 0x40191EB860000000
// CHECK-NEXT:    [[AND10:%.*]] = and i1 [[CMP]], [[CMP3]]
// CHECK-NEXT:    [[AND:%.*]] = zext i1 [[AND10]] to i32
// CHECK-NEXT:    ret i32 [[AND]]
//
int testNan(float *max, float *min, float *realMax, float *realMin) {
  float array[3] = {0. / 0., 3.14 * 2, 3.14 * 3};

  ripple_block_t BS = ripple_set_block_shape(0, 3);
  size_t v = ripple_id(BS, 0);
  float x = array[v];

  {
    *max = ripple_reducemax(0b1, x);
    *min = ripple_reducemin(0b1, x);
  }
  *realMax = array[2];
  *realMin = array[1];
  return *max == *realMax & *min == *realMin;
}

#pragma float_control(pop)

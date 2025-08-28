// REQUIRES: asserts
// REQUIRES: target-x86_64 || target-aarch64 || target=hexagon{{.*}}
// RUN: %clang -S -g -O0 -emit-llvm -fenable-ripple %s -o - | FileCheck %s
// RUN: %clang -S -g -O1 -emit-llvm -fenable-ripple %s -o -

#include <ripple.h>

// CHECK: void @fun
extern "C" void fun(size_t size, float A[size][size], float B[size][size]) {
    // CHECK: %block_X{{.*}} = alloca [8 x i{{[0-9]+}}]
    // CHECK: %block_Y{{.*}} = alloca [4 x i{{[0-9]+}}]
    // CHECK: %i = alloca i{{[0-9]+}}
    // CHECK: %ThisIs1D{{.*}}= alloca [4 x float]
    // CHECK: %ThisIsShapeShiftingFromScalarTo1DtoScalar{{.*}}= alloca [4 x float]
    // CHECK: %ThisIsScalar{{.*}}= alloca float

    ripple_block_t BS = ripple_set_block_shape(0, 8, 4);
    unsigned block_X = ripple_id(BS, 0);
    unsigned block_Y = ripple_id(BS, 1);
    unsigned size_X = ripple_get_block_size(BS, 0);
    unsigned size_Y = ripple_get_block_size(BS, 1);

    unsigned i;
    float ThisIs1D;
    // Starts as a scalar

    // CHECK: store float 0.000000e+00, ptr %ThisIsShapeShiftingFromScalarTo1DtoScalar
    float ThisIsShapeShiftingFromScalarTo1DtoScalar = 0.f;

    // Will be splat to 1D here because it might be used inside the loop w/ 1D store

    // CHECK: %[[LoadScal:[0-9]+]] = load float, ptr %ThisIsShapeShiftingFromScalarTo1DtoScalar
    // CHECK: %[[Add32:[a-zA-Z0-9_.]+]] = fadd float %[[LoadScal]], 3.200000e+01
    // CHECK: %[[Insert:[a-zA-Z0-9_.]+]] = insertelement <4 x float> poison, float %[[Add32]]
    // CHECK: %[[Shuffle:[a-zA-Z0-9_.]+]] = shufflevector <4 x float> %[[Insert]], <4 x float> poison, <4 x i{{[0-9]+}}> zeroinitializer
    // CHECK: store <4 x float> %[[Shuffle]], ptr %ThisIsShapeShiftingFromScalarTo1DtoScalar
    ThisIsShapeShiftingFromScalarTo1DtoScalar += 32;

    for (i = 0; i < size; i += size_Y)
        // 1D inside the loop because the load of A is 1D

        // CHECK: %[[LoopLoad:[a-zA-Z0-9_.]+]] = load <4 x float>, ptr %ThisIsShapeShiftingFromScalarTo1DtoScalar
        // CHECK: %[[LoopAdd:[a-zA-Z0-9_.]+]] = fadd <4 x float> %[[LoopLoad]]
        // CHECK: store <4 x float> %[[LoopAdd]], ptr %ThisIsShapeShiftingFromScalarTo1DtoScalar
        ThisIsShapeShiftingFromScalarTo1DtoScalar += A[block_Y][i];

    // This becomes 1D because of the vector branch

    // CHECK: store float 1.000000e+00, ptr %ThisIsScalar
    float ThisIsScalar = 1.;

    if (block_X < 4) {
        ThisIsScalar = 0.f;
        // This store is 2D now because it escapes the branch and
        // ThisIsShapeShiftingFromScalarTo1DTo2DTo1DToScalar is 1D in Y

        ThisIs1D = ThisIsShapeShiftingFromScalarTo1DtoScalar;
    }

    // This becomes Scalar again!

    // CHECK-LABEL: if.end
    // CHECK: [[EndBranchLoad:%.*]] = load <4 x float>, ptr %ThisIs1D
    // CHECK-NEXT:    [[DOTRIPPLE21_RIPPLE_REDUCTION_PARTIAL_MASKING:%.*]] = select <4 x i1> splat (i1 true), <4 x float> [[EndBranchLoad]], <4 x float> splat (float -0.000000e+00), !dbg [[DBG77:![0-9]+]]
    // CHECK-NEXT:    [[REDUCEFADD:%.*]] = call reassoc float @llvm.vector.reduce.fadd.v4f32(float -0.000000e+00, <4 x float> [[DOTRIPPLE21_RIPPLE_REDUCTION_PARTIAL_MASKING]])
    // CHECK: store float [[REDUCEFADD]], ptr %ThisIsScalar
    ThisIsScalar = __builtin_ripple_reduceadd_f32(0b10, ThisIs1D);

    // This splats to 2D

    // CHECK: %[[OneDReload:[a-zA-Z0-9_.]+]] = load float, ptr %ThisIsScalar
    // CHECK: %[[InsertReload:[a-zA-Z0-9_.]+]] = insertelement <4 x float> poison, float %[[OneDReload]], i{{[0-9]+}} 0
    // CHECK: %[[ShuffleReload:[a-zA-Z0-9_.]+]] = shufflevector <4 x float> %[[InsertReload]], <4 x float> poison
    ThisIsShapeShiftingFromScalarTo1DtoScalar = ThisIs1D + ThisIsScalar;

    B[block_Y][block_X] = ThisIsShapeShiftingFromScalarTo1DtoScalar;

    // 2D bcast of scalar

    // CHECK: store float 1.000000e+00, ptr %ThisIsShapeShiftingFromScalarTo1DtoScalar
    // CHECK: %[[StoreReload:[a-zA-Z0-9_.]+]] = load float, ptr %ThisIsShapeShiftingFromScalarTo1DtoScalar
    // CHECK: %[[InsertReload:[a-zA-Z0-9_.]+]] = insertelement <32 x float> poison, float %[[StoreReload]], i{{[0-9]+}} 0
    // CHECK: %[[ShuffleReload:[a-zA-Z0-9_.]+]] = shufflevector <32 x float> %[[InsertReload]], <32 x float> poison
    ThisIsShapeShiftingFromScalarTo1DtoScalar = 1.f;
    B[block_Y][block_X] += ThisIsShapeShiftingFromScalarTo1DtoScalar;
}

// CHECK-LABEL: if.then
// if (block_X < 4) {

// ThisIsOneD = 0.f; optimized away

// CHECK: %[[BranchLoad:[a-zA-Z0-9_.]+]] = load <4 x float>, ptr %ThisIsShapeShiftingFromScalarTo1DtoScalar
// CHECK: %[[SelectPoison:[a-zA-Z0-9_.]+]] = select <4 x i1> %{{.*}}, <4 x float> %[[BranchLoad]], <4 x float> poison
// CHECK: call void @llvm.masked.store.v4f32.p0(<4 x float> %[[SelectPoison]], ptr align 16 %ThisIs1D.ripple.LS.instance, <4 x i1>
// ThisBecomes2D = ThisIsShapeShiftingFromScalarTo1DTo2DTo1DToScalar;

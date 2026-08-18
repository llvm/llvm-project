// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -target-cpu gfx1250 -fcuda-is-device \
// RUN:   -emit-llvm -O1 -disable-llvm-passes %s -o - | FileCheck %s

#define __global__ __attribute__((global))

typedef float f32x8 __attribute__((ext_vector_type(8)));
typedef int i32x4 __attribute__((ext_vector_type(4)));
typedef float f32x12 __attribute__((ext_vector_type(12)));

// The intrinsic is overloaded on an integer vector: the float vector widths
// have no isel pattern of their own.
// CHECK-LABEL: @_Z4wide
// CHECK: call <8 x i32> @llvm.amdgcn.pin.vgpr.v8i32(<8 x i32> %{{.*}}, i32 256)
__global__ void wide(f32x8 *o) { *o = __builtin_amdgcn_pin_vgpr(*o, 256); }

// CHECK-LABEL: @_Z4agpr
// CHECK: call <4 x i32> @llvm.amdgcn.pin.agpr.v4i32(<4 x i32> %{{.*}}, i32 8)
__global__ void agpr(i32x4 *o) { *o = __builtin_amdgcn_pin_agpr(*o, 8); }

// CHECK-LABEL: @_Z6scalar
// CHECK: call i32 @llvm.amdgcn.pin.vgpr.i32(i32 %{{.*}}, i32 12)
__global__ void scalar(int *o) { *o = __builtin_amdgcn_pin_vgpr(*o, 12); }

// A width with no pattern of its own is split into ones that have, each
// asking for its own part of the tuple.
// CHECK-LABEL: @_Z7unusual
// CHECK: call <8 x i32> @llvm.amdgcn.pin.vgpr.v8i32(<8 x i32> %{{.*}}, i32 256)
// CHECK: call <4 x i32> @llvm.amdgcn.pin.vgpr.v4i32(<4 x i32> %{{.*}}, i32 264)
__global__ void unusual(f32x12 *o) { *o = __builtin_amdgcn_pin_vgpr(*o, 256); }

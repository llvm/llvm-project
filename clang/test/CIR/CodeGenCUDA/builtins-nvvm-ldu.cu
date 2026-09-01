#include "Inputs/cuda.h"

// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -target-cpu sm_80 -x cuda \
// RUN:            -fcuda-is-device -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s

// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -target-cpu sm_80 -x cuda \
// RUN:            -fcuda-is-device -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s

// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -target-cpu sm_80 -x cuda \
// RUN:            -fcuda-is-device -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// FIXME: CIR doesn't propagate the 'contract' fast-math flag to LLVM IR calls
// yet, so the floating-point LLVM check lines use {{.*}} to tolerate the
// difference between CIR (no flags) and classic codegen ('contract').

typedef char char2 __attribute__((ext_vector_type(2)));
typedef unsigned char uchar2 __attribute__((ext_vector_type(2)));
typedef signed char schar2 __attribute__((ext_vector_type(2)));
typedef char char4 __attribute__((ext_vector_type(4)));
typedef unsigned char uchar4 __attribute__((ext_vector_type(4)));
typedef signed char schar4 __attribute__((ext_vector_type(4)));
typedef short short2 __attribute__((ext_vector_type(2)));
typedef unsigned short ushort2 __attribute__((ext_vector_type(2)));
typedef short short4 __attribute__((ext_vector_type(4)));
typedef unsigned short ushort4 __attribute__((ext_vector_type(4)));
typedef int int2 __attribute__((ext_vector_type(2)));
typedef unsigned int uint2 __attribute__((ext_vector_type(2)));
typedef int int4 __attribute__((ext_vector_type(4)));
typedef unsigned int uint4 __attribute__((ext_vector_type(4)));
typedef long long2 __attribute__((ext_vector_type(2)));
typedef unsigned long ulong2 __attribute__((ext_vector_type(2)));
typedef long long longlong2 __attribute__((ext_vector_type(2)));
typedef unsigned long long ulonglong2 __attribute__((ext_vector_type(2)));
typedef float float2 __attribute__((ext_vector_type(2)));
typedef float float4 __attribute__((ext_vector_type(4)));
typedef double double2 __attribute__((ext_vector_type(2)));

// CIR-LABEL: @_Z8nvvm_lduPKv
// LLVM-LABEL: @_Z8nvvm_lduPKv
__device__ void nvvm_ldu(const void *p) {
  // CIR: cir.call_llvm_intrinsic "nvvm.ldu.global.i" {{.*}} : (!cir.ptr<!s8i>, !s32i) -> !s8i
  // CIR: cir.call_llvm_intrinsic "nvvm.ldu.global.i" {{.*}} : (!cir.ptr<!u8i>, !s32i) -> !u8i
  // CIR: cir.call_llvm_intrinsic "nvvm.ldu.global.i" {{.*}} : (!cir.ptr<!s8i>, !s32i) -> !s8i
  // LLVM: call i8 @llvm.nvvm.ldu.global.i.i8.p0(ptr {{%[0-9]+}}, i32 1)
  // LLVM: call i8 @llvm.nvvm.ldu.global.i.i8.p0(ptr {{%[0-9]+}}, i32 1)
  // LLVM: call i8 @llvm.nvvm.ldu.global.i.i8.p0(ptr {{%[0-9]+}}, i32 1)
  __nvvm_ldu_c((const char *)p);
  __nvvm_ldu_uc((const unsigned char *)p);
  __nvvm_ldu_sc((const signed char *)p);

  // CIR: cir.call_llvm_intrinsic "nvvm.ldu.global.i" {{.*}} : (!cir.ptr<!s16i>, !s32i) -> !s16i
  // CIR: cir.call_llvm_intrinsic "nvvm.ldu.global.i" {{.*}} : (!cir.ptr<!u16i>, !s32i) -> !u16i
  // LLVM: call i16 @llvm.nvvm.ldu.global.i.i16.p0(ptr {{%[0-9]+}}, i32 2)
  // LLVM: call i16 @llvm.nvvm.ldu.global.i.i16.p0(ptr {{%[0-9]+}}, i32 2)
  __nvvm_ldu_s((const short *)p);
  __nvvm_ldu_us((const unsigned short *)p);

  // CIR: cir.call_llvm_intrinsic "nvvm.ldu.global.i" {{.*}} : (!cir.ptr<!s32i>, !s32i) -> !s32i
  // CIR: cir.call_llvm_intrinsic "nvvm.ldu.global.i" {{.*}} : (!cir.ptr<!u32i>, !s32i) -> !u32i
  // LLVM: call i32 @llvm.nvvm.ldu.global.i.i32.p0(ptr {{%[0-9]+}}, i32 4)
  // LLVM: call i32 @llvm.nvvm.ldu.global.i.i32.p0(ptr {{%[0-9]+}}, i32 4)
  __nvvm_ldu_i((const int *)p);
  __nvvm_ldu_ui((const unsigned int *)p);

  // CIR: cir.call_llvm_intrinsic "nvvm.ldu.global.i" {{.*}} : (!cir.ptr<!s64i>, !s32i) -> !s64i
  // CIR: cir.call_llvm_intrinsic "nvvm.ldu.global.i" {{.*}} : (!cir.ptr<!u64i>, !s32i) -> !u64i
  // LLVM: call i64 @llvm.nvvm.ldu.global.i.i64.p0(ptr {{%[0-9]+}}, i32 8)
  // LLVM: call i64 @llvm.nvvm.ldu.global.i.i64.p0(ptr {{%[0-9]+}}, i32 8)
  __nvvm_ldu_l((const long *)p);
  __nvvm_ldu_ul((const unsigned long *)p);

  // CIR: cir.call_llvm_intrinsic "nvvm.ldu.global.f" {{.*}} : (!cir.ptr<!cir.float>, !s32i) -> !cir.float
  // LLVM: call {{.*}}float @llvm.nvvm.ldu.global.f.f32.p0(ptr {{%[0-9]+}}, i32 4)
  __nvvm_ldu_f((const float *)p);
  // CIR: cir.call_llvm_intrinsic "nvvm.ldu.global.f" {{.*}} : (!cir.ptr<!cir.double>, !s32i) -> !cir.double
  // LLVM: call {{.*}}double @llvm.nvvm.ldu.global.f.f64.p0(ptr {{%[0-9]+}}, i32 8)
  __nvvm_ldu_d((const double *)p);

  // CIR: cir.call_llvm_intrinsic "nvvm.ldu.global.i" {{.*}} : (!cir.ptr<!cir.vector<2 x !s8i>>, !s32i) -> !cir.vector<2 x !s8i>
  // CIR: cir.call_llvm_intrinsic "nvvm.ldu.global.i" {{.*}} : (!cir.ptr<!cir.vector<2 x !u8i>>, !s32i) -> !cir.vector<2 x !u8i>
  // CIR: cir.call_llvm_intrinsic "nvvm.ldu.global.i" {{.*}} : (!cir.ptr<!cir.vector<2 x !s8i>>, !s32i) -> !cir.vector<2 x !s8i>
  // LLVM: call <2 x i8> @llvm.nvvm.ldu.global.i.v2i8.p0(ptr {{%[0-9]+}}, i32 2)
  // LLVM: call <2 x i8> @llvm.nvvm.ldu.global.i.v2i8.p0(ptr {{%[0-9]+}}, i32 2)
  // LLVM: call <2 x i8> @llvm.nvvm.ldu.global.i.v2i8.p0(ptr {{%[0-9]+}}, i32 2)
  __nvvm_ldu_c2((const char2 *)p);
  __nvvm_ldu_uc2((const uchar2 *)p);
  __nvvm_ldu_sc2((const schar2 *)p);

  // CIR: cir.call_llvm_intrinsic "nvvm.ldu.global.i" {{.*}} : (!cir.ptr<!cir.vector<4 x !s8i>>, !s32i) -> !cir.vector<4 x !s8i>
  // CIR: cir.call_llvm_intrinsic "nvvm.ldu.global.i" {{.*}} : (!cir.ptr<!cir.vector<4 x !u8i>>, !s32i) -> !cir.vector<4 x !u8i>
  // CIR: cir.call_llvm_intrinsic "nvvm.ldu.global.i" {{.*}} : (!cir.ptr<!cir.vector<4 x !s8i>>, !s32i) -> !cir.vector<4 x !s8i>
  // LLVM: call <4 x i8> @llvm.nvvm.ldu.global.i.v4i8.p0(ptr {{%[0-9]+}}, i32 4)
  // LLVM: call <4 x i8> @llvm.nvvm.ldu.global.i.v4i8.p0(ptr {{%[0-9]+}}, i32 4)
  // LLVM: call <4 x i8> @llvm.nvvm.ldu.global.i.v4i8.p0(ptr {{%[0-9]+}}, i32 4)
  __nvvm_ldu_c4((const char4 *)p);
  __nvvm_ldu_uc4((const uchar4 *)p);
  __nvvm_ldu_sc4((const schar4 *)p);

  // CIR: cir.call_llvm_intrinsic "nvvm.ldu.global.i" {{.*}} : (!cir.ptr<!cir.vector<2 x !s16i>>, !s32i) -> !cir.vector<2 x !s16i>
  // CIR: cir.call_llvm_intrinsic "nvvm.ldu.global.i" {{.*}} : (!cir.ptr<!cir.vector<2 x !u16i>>, !s32i) -> !cir.vector<2 x !u16i>
  // LLVM: call <2 x i16> @llvm.nvvm.ldu.global.i.v2i16.p0(ptr {{%[0-9]+}}, i32 4)
  // LLVM: call <2 x i16> @llvm.nvvm.ldu.global.i.v2i16.p0(ptr {{%[0-9]+}}, i32 4)
  __nvvm_ldu_s2((const short2 *)p);
  __nvvm_ldu_us2((const ushort2 *)p);

  // CIR: cir.call_llvm_intrinsic "nvvm.ldu.global.i" {{.*}} : (!cir.ptr<!cir.vector<4 x !s16i>>, !s32i) -> !cir.vector<4 x !s16i>
  // CIR: cir.call_llvm_intrinsic "nvvm.ldu.global.i" {{.*}} : (!cir.ptr<!cir.vector<4 x !u16i>>, !s32i) -> !cir.vector<4 x !u16i>
  // LLVM: call <4 x i16> @llvm.nvvm.ldu.global.i.v4i16.p0(ptr {{%[0-9]+}}, i32 8)
  // LLVM: call <4 x i16> @llvm.nvvm.ldu.global.i.v4i16.p0(ptr {{%[0-9]+}}, i32 8)
  __nvvm_ldu_s4((const short4 *)p);
  __nvvm_ldu_us4((const ushort4 *)p);

  // CIR: cir.call_llvm_intrinsic "nvvm.ldu.global.i" {{.*}} : (!cir.ptr<!cir.vector<2 x !s32i>>, !s32i) -> !cir.vector<2 x !s32i>
  // CIR: cir.call_llvm_intrinsic "nvvm.ldu.global.i" {{.*}} : (!cir.ptr<!cir.vector<2 x !u32i>>, !s32i) -> !cir.vector<2 x !u32i>
  // LLVM: call <2 x i32> @llvm.nvvm.ldu.global.i.v2i32.p0(ptr {{%[0-9]+}}, i32 8)
  // LLVM: call <2 x i32> @llvm.nvvm.ldu.global.i.v2i32.p0(ptr {{%[0-9]+}}, i32 8)
  __nvvm_ldu_i2((const int2 *)p);
  __nvvm_ldu_ui2((const uint2 *)p);

  // CIR: cir.call_llvm_intrinsic "nvvm.ldu.global.i" {{.*}} : (!cir.ptr<!cir.vector<4 x !s32i>>, !s32i) -> !cir.vector<4 x !s32i>
  // CIR: cir.call_llvm_intrinsic "nvvm.ldu.global.i" {{.*}} : (!cir.ptr<!cir.vector<4 x !u32i>>, !s32i) -> !cir.vector<4 x !u32i>
  // LLVM: call <4 x i32> @llvm.nvvm.ldu.global.i.v4i32.p0(ptr {{%[0-9]+}}, i32 16)
  // LLVM: call <4 x i32> @llvm.nvvm.ldu.global.i.v4i32.p0(ptr {{%[0-9]+}}, i32 16)
  __nvvm_ldu_i4((const int4 *)p);
  __nvvm_ldu_ui4((const uint4 *)p);

  // CIR: cir.call_llvm_intrinsic "nvvm.ldu.global.i" {{.*}} : (!cir.ptr<!cir.vector<2 x !s64i>>, !s32i) -> !cir.vector<2 x !s64i>
  // CIR: cir.call_llvm_intrinsic "nvvm.ldu.global.i" {{.*}} : (!cir.ptr<!cir.vector<2 x !u64i>>, !s32i) -> !cir.vector<2 x !u64i>
  // LLVM: call <2 x i64> @llvm.nvvm.ldu.global.i.v2i64.p0(ptr {{%[0-9]+}}, i32 16)
  // LLVM: call <2 x i64> @llvm.nvvm.ldu.global.i.v2i64.p0(ptr {{%[0-9]+}}, i32 16)
  __nvvm_ldu_l2((const long2 *)p);
  __nvvm_ldu_ul2((const ulong2 *)p);

  // CIR: cir.call_llvm_intrinsic "nvvm.ldu.global.i" {{.*}} : (!cir.ptr<!cir.vector<2 x !s64i>>, !s32i) -> !cir.vector<2 x !s64i>
  // CIR: cir.call_llvm_intrinsic "nvvm.ldu.global.i" {{.*}} : (!cir.ptr<!cir.vector<2 x !u64i>>, !s32i) -> !cir.vector<2 x !u64i>
  // LLVM: call <2 x i64> @llvm.nvvm.ldu.global.i.v2i64.p0(ptr {{%[0-9]+}}, i32 16)
  // LLVM: call <2 x i64> @llvm.nvvm.ldu.global.i.v2i64.p0(ptr {{%[0-9]+}}, i32 16)
  __nvvm_ldu_ll2((const longlong2 *)p);
  __nvvm_ldu_ull2((const ulonglong2 *)p);

  // CIR: cir.call_llvm_intrinsic "nvvm.ldu.global.f" {{.*}} : (!cir.ptr<!cir.vector<2 x !cir.float>>, !s32i) -> !cir.vector<2 x !cir.float>
  // LLVM: call {{.*}}<2 x float> @llvm.nvvm.ldu.global.f.v2f32.p0(ptr {{%[0-9]+}}, i32 8)
  __nvvm_ldu_f2((const float2 *)p);

  // CIR: cir.call_llvm_intrinsic "nvvm.ldu.global.f" {{.*}} : (!cir.ptr<!cir.vector<4 x !cir.float>>, !s32i) -> !cir.vector<4 x !cir.float>
  // LLVM: call {{.*}}<4 x float> @llvm.nvvm.ldu.global.f.v4f32.p0(ptr {{%[0-9]+}}, i32 16)
  __nvvm_ldu_f4((const float4 *)p);

  // CIR: cir.call_llvm_intrinsic "nvvm.ldu.global.f" {{.*}} : (!cir.ptr<!cir.vector<2 x !cir.double>>, !s32i) -> !cir.vector<2 x !cir.double>
  // LLVM: call {{.*}}<2 x double> @llvm.nvvm.ldu.global.f.v2f64.p0(ptr {{%[0-9]+}}, i32 16)
  __nvvm_ldu_d2((const double2 *)p);
}

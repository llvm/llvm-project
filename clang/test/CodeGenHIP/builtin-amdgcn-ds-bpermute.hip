// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -target-cpu gfx1200 -aux-triple x86_64-pc-linux-gnu \
// RUN:    -fcuda-is-device -emit-llvm -o - %s | FileCheck %s

typedef short short2 __attribute__((vector_size(4)));

#define __device__ __attribute__((device))

// CHECK-LABEL: define{{.*}}@test_index_i32
// CHECK: call i32 @llvm.amdgcn.ds.bpermute(i32 %0, i32 %1)
extern "C" __device__ int test_index_i32(int a, int c) {
   return __builtin_amdgcn_ds_bpermute(a, c);
}

// CHECK-LABEL: define{{.*}}@test_index_long
// CHECK: [[TRUNC:%.*]] = trunc i64 %0 to i32
// CHECK: call i32 @llvm.amdgcn.ds.bpermute(i32 [[TRUNC]], i32 %1)
extern "C" __device__ int test_index_long(long a, int c) {
   return __builtin_amdgcn_ds_bpermute(a, c);
}

// CHECK-LABEL: define{{.*}}@test_index_short
// CHECK: [[EXT:%.*]] = sext i16 %0 to i32
// CHECK: call i32 @llvm.amdgcn.ds.bpermute(i32 [[EXT]], i32 %1)
extern "C" __device__ int test_index_short(short a, int c) {
   return __builtin_amdgcn_ds_bpermute(a, c);
}

// CHECK-LABEL: define{{.*}}@test_index_float
// CHECK: [[CONV:%.*]] = fptosi float %0 to i32
// CHECK: call i32 @llvm.amdgcn.ds.bpermute(i32 [[CONV]], i32 %1)
extern "C" __device__ int test_index_float(float a, int c) {
   return __builtin_amdgcn_ds_bpermute(a, c);
}

// CHECK-LABEL: define{{.*}}@test_index_struct
// CHECK: [[CALL:%.*]] = call noundef i32 @_ZNK11ConvertiblecviEv(
// CHECK: call i32 @llvm.amdgcn.ds.bpermute(i32 [[CALL]], i32 %{{[0-9]+}})
struct Convertible {
  int value;
  __device__ operator int() const { return value; }
};

extern "C" __device__ int test_index_struct(Convertible a, int c) {
   return __builtin_amdgcn_ds_bpermute(a, c);
}

template<typename T>
__device__ int test_template(T a, int c) {
   return __builtin_amdgcn_ds_bpermute(a, c);
}

// CHECK-LABEL: define{{.*}}@test_template_short
// CHECK: [[EXT:%.*]] = sext i16 %0 to i32
// CHECK: call i32 @llvm.amdgcn.ds.bpermute(i32 [[EXT]], i32 %1)
extern "C" __device__ int test_template_short(short a, int c) {
   return test_template<short>(a, c);
}

// CHECK-LABEL: define{{.*}}@test_source_float
// CHECK: [[BITCAST:%.*]] = bitcast float %1 to i32
// CHECK: [[CALL:%.*]] = call i32 @llvm.amdgcn.ds.bpermute(i32 %0, i32 [[BITCAST]])
// CHECK: [[RESULT:%.*]] = bitcast i32 [[CALL]] to float
// CHECK: ret float [[RESULT]]
extern "C" __device__ float test_source_float(int a, float c) {
   return __builtin_amdgcn_ds_bpermute(a, c);
}

// CHECK-LABEL: define{{.*}}@test_source_v2i16
// CHECK: [[BITCAST:%.*]] = bitcast <2 x i16> %1 to i32
// CHECK: [[CALL:%.*]] = call i32 @llvm.amdgcn.ds.bpermute(i32 %0, i32 [[BITCAST]])
// CHECK: [[RESULT:%.*]] = bitcast i32 [[CALL]] to <2 x i16>
// CHECK: ret <2 x i16> [[RESULT]]
extern "C" __device__ short2 test_source_v2i16(int a, short2 c) {
   return __builtin_amdgcn_ds_bpermute(a, c);
}

// CHECK-LABEL: define{{.*}}@test_source_float16
// CHECK: [[BITCAST:%.*]] = bitcast half %1 to i16
// CHECK: [[ZEXT:%.*]] = zext i16 [[BITCAST]] to i32
// CHECK: [[CALL:%.*]] = call i32 @llvm.amdgcn.ds.bpermute(i32 %0, i32 [[ZEXT]])
// CHECK: [[TRUNC:%.*]] = trunc i32 [[CALL]] to i16
// CHECK: [[RESULT:%.*]] = bitcast i16 [[TRUNC]] to half
// CHECK: ret half [[RESULT]]
extern "C" __device__ _Float16 test_source_float16(int a, _Float16 c) {
   return __builtin_amdgcn_ds_bpermute(a, c);
}

// CHECK-LABEL: define{{.*}}@test_source_short
// CHECK: [[SEXT:%.*]] = sext i16 %1 to i32
// CHECK: [[CALL:%.*]] = call i32 @llvm.amdgcn.ds.bpermute(i32 %0, i32 [[SEXT]])
// CHECK: [[TRUNC:%.*]] = trunc i32 [[CALL]] to i16
// CHECK: ret i16 [[TRUNC]]
extern "C" __device__ short test_source_short(int a, short c) {
   return __builtin_amdgcn_ds_bpermute(a, c);
}

// CHECK-LABEL: define{{.*}}@test_source_ushort
// CHECK: [[ZEXT:%.*]] = zext i16 %1 to i32
// CHECK: [[CALL:%.*]] = call i32 @llvm.amdgcn.ds.bpermute(i32 %0, i32 [[ZEXT]])
// CHECK: [[TRUNC:%.*]] = trunc i32 [[CALL]] to i16
// CHECK: ret i16 [[TRUNC]]
extern "C" __device__ unsigned short test_source_ushort(int a, unsigned short c) {
   return __builtin_amdgcn_ds_bpermute(a, c);
}

// CHECK-LABEL: define{{.*}}@test_source_long
// CHECK: [[TRUNC:%.*]] = trunc i64 %1 to i32
// CHECK: [[CALL:%.*]] = call i32 @llvm.amdgcn.ds.bpermute(i32 %0, i32 [[TRUNC]])
// CHECK: [[SEXT:%.*]] = sext i32 [[CALL]] to i64
// CHECK: ret i64 [[SEXT]]
extern "C" __device__ long test_source_long(int a, long c) {
   return __builtin_amdgcn_ds_bpermute(a, c);
}

// CHECK-LABEL: define{{.*}}@test_source_ulong
// CHECK: [[TRUNC:%.*]] = trunc i64 %1 to i32
// CHECK: [[CALL:%.*]] = call i32 @llvm.amdgcn.ds.bpermute(i32 %0, i32 [[TRUNC]])
// CHECK: [[ZEXT:%.*]] = zext i32 [[CALL]] to i64
// CHECK: ret i64 [[ZEXT]]
extern "C" __device__ unsigned long test_source_ulong(int a, unsigned long c) {
   return __builtin_amdgcn_ds_bpermute(a, c);
}

// CHECK-LABEL: define{{.*}}@test_source_double
// CHECK: [[FPTRUNC:%.*]] = fptrunc contract double %1 to float
// CHECK: [[BITCAST:%.*]] = bitcast float [[FPTRUNC]] to i32
// CHECK: [[CALL:%.*]] = call i32 @llvm.amdgcn.ds.bpermute(i32 %0, i32 [[BITCAST]])
// CHECK: [[BITCAST2:%.*]] = bitcast i32 [[CALL]] to float
// CHECK: [[FPEXT:%.*]] = fpext contract float [[BITCAST2]] to double
// CHECK: ret double [[FPEXT]]
extern "C" __device__ double test_source_double(int a, double c) {
   return __builtin_amdgcn_ds_bpermute(a, c);
}

// CHECK-LABEL: define{{.*}}@test_template_float_src
// CHECK: [[BITCAST:%.*]] = bitcast float %1 to i32
// CHECK: [[CALL:%.*]] = call i32 @llvm.amdgcn.ds.bpermute(i32 %0, i32 [[BITCAST]])
// CHECK: [[RESULT:%.*]] = bitcast i32 [[CALL]] to float
// CHECK: ret float [[RESULT]]
template<typename T>
__device__ T test_template_src(int a, T c) {
   return __builtin_amdgcn_ds_bpermute(a, c);
}

extern "C" __device__ float test_template_float_src(int a, float c) {
   return test_template_src<float>(a, c);
}

// CHECK-LABEL: define{{.*}}@test_source_struct_float
// CHECK: [[CALL:%.*]] = call {{.*}} float @_ZNK16FloatConvertiblecvfEv(
// CHECK: [[CONV:%.*]] = bitcast float [[CALL]] to i32
// CHECK: [[RESULT:%.*]] = call i32 @llvm.amdgcn.ds.bpermute(i32 %0, i32 [[CONV]])
// CHECK: [[CONV2:%.*]] = bitcast i32 [[RESULT]] to float
// CHECK: ret float [[CONV2]]
struct FloatConvertible {
  float value;
  __device__ operator float() const { return value; }
};

extern "C" __device__ float test_source_struct_float(int a, FloatConvertible c) {
   return __builtin_amdgcn_ds_bpermute(a, c);
}

// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -target-cpu gfx1200 -aux-triple x86_64-pc-linux-gnu \
// RUN:    -fcuda-is-device -emit-llvm -o - %s | FileCheck %s

typedef short short2 __attribute__((vector_size(4)));
typedef double double2 __attribute__((ext_vector_type(2)));

struct Inner { short a; char b; };
struct Outer { int x; struct Inner y; char z; };
struct __attribute__((packed)) Three { char a, b, c; };

union U { int i; char c; };

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
// CHECK: [[ZEXT:%.*]] = zext i16 %1 to i32
// CHECK: [[CALL:%.*]] = call i32 @llvm.amdgcn.ds.bpermute(i32 %0, i32 [[ZEXT]])
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
// CHECK: [[LO:%.*]] = trunc i64 %1 to i32
// CHECK: [[RLO:%.*]] = call i32 @llvm.amdgcn.ds.bpermute(i32 %0, i32 [[LO]])
// CHECK: [[V0:%.*]] = zext i32 [[RLO]] to i64
// CHECK: [[SHR:%.*]] = lshr i64 %1, 32
// CHECK: [[HI:%.*]] = trunc i64 [[SHR]] to i32
// CHECK: [[RHI:%.*]] = call i32 @llvm.amdgcn.ds.bpermute(i32 %0, i32 [[HI]])
// CHECK: [[V1:%.*]] = zext i32 [[RHI]] to i64
// CHECK: [[V1S:%.*]] = shl i64 [[V1]], 32
// CHECK: [[RES:%.*]] = or i64 [[V0]], [[V1S]]
// CHECK: ret i64 [[RES]]
extern "C" __device__ long test_source_long(int a, long c) {
   return __builtin_amdgcn_ds_bpermute(a, c);
}

// CHECK-LABEL: define{{.*}}@test_source_ulong
// CHECK: [[LO:%.*]] = trunc i64 %1 to i32
// CHECK: [[RLO:%.*]] = call i32 @llvm.amdgcn.ds.bpermute(i32 %0, i32 [[LO]])
// CHECK: [[V0:%.*]] = zext i32 [[RLO]] to i64
// CHECK: [[SHR:%.*]] = lshr i64 %1, 32
// CHECK: [[HI:%.*]] = trunc i64 [[SHR]] to i32
// CHECK: [[RHI:%.*]] = call i32 @llvm.amdgcn.ds.bpermute(i32 %0, i32 [[HI]])
// CHECK: [[V1:%.*]] = zext i32 [[RHI]] to i64
// CHECK: [[V1S:%.*]] = shl i64 [[V1]], 32
// CHECK: [[RES:%.*]] = or i64 [[V0]], [[V1S]]
// CHECK: ret i64 [[RES]]
extern "C" __device__ unsigned long test_source_ulong(int a, unsigned long c) {
   return __builtin_amdgcn_ds_bpermute(a, c);
}

// CHECK-LABEL: define{{.*}}@test_source_double
// CHECK: [[BC:%.*]] = bitcast double %1 to i64
// CHECK: [[LO:%.*]] = trunc i64 [[BC]] to i32
// CHECK: [[RLO:%.*]] = call i32 @llvm.amdgcn.ds.bpermute(i32 %0, i32 [[LO]])
// CHECK: [[V0:%.*]] = zext i32 [[RLO]] to i64
// CHECK: [[SHR:%.*]] = lshr i64 [[BC]], 32
// CHECK: [[HI:%.*]] = trunc i64 [[SHR]] to i32
// CHECK: [[RHI:%.*]] = call i32 @llvm.amdgcn.ds.bpermute(i32 %0, i32 [[HI]])
// CHECK: [[V1:%.*]] = zext i32 [[RHI]] to i64
// CHECK: [[V1S:%.*]] = shl i64 [[V1]], 32
// CHECK: [[RES64:%.*]] = or i64 [[V0]], [[V1S]]
// CHECK: [[RES:%.*]] = bitcast i64 [[RES64]] to double
// CHECK: ret double [[RES]]
extern "C" __device__ double test_source_double(int a, double c) {
   return __builtin_amdgcn_ds_bpermute(a, c);
}

// CHECK-LABEL: define{{.*}}@test_source_ptr
// CHECK: [[P2I:%.*]] = ptrtoint ptr %1 to i64
// CHECK: [[LO:%.*]] = trunc i64 [[P2I]] to i32
// CHECK: [[RLO:%.*]] = call i32 @llvm.amdgcn.ds.bpermute(i32 %0, i32 [[LO]])
// CHECK: [[V0:%.*]] = zext i32 [[RLO]] to i64
// CHECK: [[SHR:%.*]] = lshr i64 [[P2I]], 32
// CHECK: [[HI:%.*]] = trunc i64 [[SHR]] to i32
// CHECK: [[RHI:%.*]] = call i32 @llvm.amdgcn.ds.bpermute(i32 %0, i32 [[HI]])
// CHECK: [[V1:%.*]] = zext i32 [[RHI]] to i64
// CHECK: [[V1S:%.*]] = shl i64 [[V1]], 32
// CHECK: [[I64RES:%.*]] = or i64 [[V0]], [[V1S]]
// CHECK: [[PRES:%.*]] = inttoptr i64 [[I64RES]] to ptr
// CHECK: ret ptr [[PRES]]
extern "C" __device__ void* test_source_ptr(int a, void* c) {
  return __builtin_amdgcn_ds_bpermute(a, c);
}

// Address space 1 (global) pointer: 64-bit, split into 2 words
// CHECK-LABEL: define{{.*}}@test_source_global_ptr
// CHECK: [[P2I:%.*]] = ptrtoint ptr addrspace(1) %1 to i64
// CHECK: [[LO:%.*]] = trunc i64 [[P2I]] to i32
// CHECK: [[RLO:%.*]] = call i32 @llvm.amdgcn.ds.bpermute(i32 %0, i32 [[LO]])
// CHECK: [[V0:%.*]] = zext i32 [[RLO]] to i64
// CHECK: [[SHR:%.*]] = lshr i64 [[P2I]], 32
// CHECK: [[HI:%.*]] = trunc i64 [[SHR]] to i32
// CHECK: [[RHI:%.*]] = call i32 @llvm.amdgcn.ds.bpermute(i32 %0, i32 [[HI]])
// CHECK: [[V1:%.*]] = zext i32 [[RHI]] to i64
// CHECK: [[V1S:%.*]] = shl i64 [[V1]], 32
// CHECK: [[I64RES:%.*]] = or i64 [[V0]], [[V1S]]
// CHECK: [[PRES:%.*]] = inttoptr i64 [[I64RES]] to ptr addrspace(1)
// CHECK: ret ptr addrspace(1) [[PRES]]
extern "C" __device__ int __attribute__((address_space(1)))* test_source_global_ptr(int a, int __attribute__((address_space(1)))* c) {
  return __builtin_amdgcn_ds_bpermute(a, c);
}

// Address space 3 (LDS) pointer: 32-bit, single bpermute
// CHECK-LABEL: define{{.*}}@test_source_local_ptr
// CHECK: [[P2I:%.*]] = ptrtoint ptr addrspace(3) %1 to i32
// CHECK: [[PERM:%.*]] = call i32 @llvm.amdgcn.ds.bpermute(i32 %0, i32 [[P2I]])
// CHECK: [[PRES:%.*]] = inttoptr i32 [[PERM]] to ptr addrspace(3)
// CHECK: ret ptr addrspace(3) [[PRES]]
extern "C" __device__ int __attribute__((address_space(3)))* test_source_local_ptr(int a, int __attribute__((address_space(3)))* c) {
  return __builtin_amdgcn_ds_bpermute(a, c);
}

// CHECK-LABEL: define{{.*}}@test_source_double2
// CHECK: [[BC:%.*]] = bitcast <2 x double> %1 to i128
// CHECK: [[W0:%.*]] = trunc i128 [[BC]] to i32
// CHECK: [[R0:%.*]] = call i32 @llvm.amdgcn.ds.bpermute(i32 %0, i32 [[W0]])
// CHECK: [[Z0:%.*]] = zext i32 [[R0]] to i128
// CHECK: [[SHR1:%.*]] = lshr i128 [[BC]], 32
// CHECK: [[W1:%.*]] = trunc i128 [[SHR1]] to i32
// CHECK: [[R1:%.*]] = call i32 @llvm.amdgcn.ds.bpermute(i32 %0, i32 [[W1]])
// CHECK: [[Z1:%.*]] = zext i32 [[R1]] to i128
// CHECK: [[S1:%.*]] = shl i128 [[Z1]], 32
// CHECK: [[OR1:%.*]] = or i128 [[Z0]], [[S1]]
// CHECK: [[SHR2:%.*]] = lshr i128 [[BC]], 64
// CHECK: [[W2:%.*]] = trunc i128 [[SHR2]] to i32
// CHECK: [[R2:%.*]] = call i32 @llvm.amdgcn.ds.bpermute(i32 %0, i32 [[W2]])
// CHECK: [[Z2:%.*]] = zext i32 [[R2]] to i128
// CHECK: [[S2:%.*]] = shl i128 [[Z2]], 64
// CHECK: [[OR2:%.*]] = or i128 [[OR1]], [[S2]]
// CHECK: [[SHR3:%.*]] = lshr i128 [[BC]], 96
// CHECK: [[W3:%.*]] = trunc i128 [[SHR3]] to i32
// CHECK: [[R3:%.*]] = call i32 @llvm.amdgcn.ds.bpermute(i32 %0, i32 [[W3]])
// CHECK: [[Z3:%.*]] = zext i32 [[R3]] to i128
// CHECK: [[S3:%.*]] = shl i128 [[Z3]], 96
// CHECK: [[OR3:%.*]] = or i128 [[OR2]], [[S3]]
// CHECK: [[RES:%.*]] = bitcast i128 [[OR3]] to <2 x double>
// CHECK: ret <2 x double> [[RES]]
extern "C" __device__ double2 test_source_double2(int a, double2 c) {
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

// CHECK-LABEL: define{{.*}}@test_source_nested(
// Materialize src aggregate, copy to bperm.src, load as i96
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %bperm.src.ascast, ptr align 4 %src1, i64 12, i1 false)
// CHECK:   %[[SRCINT:.*]] = load i96, ptr %bperm.src.ascast, align 4
// Word 0
// CHECK:   %[[W0:.*]] = trunc i96 %[[SRCINT]] to i32
// CHECK:   %[[R0:.*]] = call i32 @llvm.amdgcn.ds.bpermute(i32 %{{[0-9]+}}, i32 %[[W0]])
// CHECK:   %[[Z0:.*]] = zext i32 %[[R0]] to i96
// Word 1
// CHECK:   %[[SHR1:.*]] = lshr i96 %[[SRCINT]], 32
// CHECK:   %[[W1:.*]] = trunc i96 %[[SHR1]] to i32
// CHECK:   %[[R1:.*]] = call i32 @llvm.amdgcn.ds.bpermute(i32 %{{[0-9]+}}, i32 %[[W1]])
// CHECK:   %[[Z1:.*]] = zext i32 %[[R1]] to i96
// CHECK:   %[[S1:.*]] = shl i96 %[[Z1]], 32
// CHECK:   %[[OR1:.*]] = or i96 %[[Z0]], %[[S1]]
// Word 2
// CHECK:   %[[SHR2:.*]] = lshr i96 %[[SRCINT]], 64
// CHECK:   %[[W2:.*]] = trunc i96 %[[SHR2]] to i32
// CHECK:   %[[R2:.*]] = call i32 @llvm.amdgcn.ds.bpermute(i32 %{{[0-9]+}}, i32 %[[W2]])
// CHECK:   %[[Z2:.*]] = zext i32 %[[R2]] to i96
// CHECK:   %[[S2:.*]] = shl i96 %[[Z2]], 64
// CHECK:   %[[OR2:.*]] = or i96 %[[OR1]], %[[S2]]
// Store result and return
// CHECK:   store i96 %[[OR2]], ptr addrspace(5) %retval, align 4
// CHECK:   %[[RES:.*]] = load %struct.Outer, ptr addrspace(5) %retval, align 4
// CHECK:   ret %struct.Outer %[[RES]]
extern "C" __device__ Outer test_source_nested(int idx, Outer src) {
  return __builtin_amdgcn_ds_bpermute(idx, src);
}

// CHECK-LABEL: define{{.*}}@test_source_packed3(
// 3-byte packed struct: load as i24, zext to i32, permute, trunc back to i24
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %bperm.src.ascast, ptr align 1 %src1, i64 3, i1 false)
// CHECK:   %[[SRCINT:.*]] = load i24, ptr %bperm.src.ascast, align 1
// CHECK:   %[[ZEXT:.*]] = zext i24 %[[SRCINT]] to i32
// CHECK:   %[[PERM:.*]] = call i32 @llvm.amdgcn.ds.bpermute(i32 %{{[0-9]+}}, i32 %[[ZEXT]])
// CHECK:   %[[TRUNC:.*]] = trunc i32 %[[PERM]] to i24
// CHECK:   store i24 %[[TRUNC]], ptr addrspace(5) %retval, align 1
extern "C" __device__ Three test_source_packed3(int idx, Three src) {
  return __builtin_amdgcn_ds_bpermute(idx, src);
}

// CHECK-LABEL: define{{.*}}@test_source_union(
// Copy to bperm.src, load as i32, permute, store back
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %bperm.src.ascast, ptr align 4 %src1, i64 4, i1 false)
// CHECK:   %[[SRCINT:.*]] = load i32, ptr %bperm.src.ascast, align 4
// CHECK:   %[[PERM:.*]] = call i32 @llvm.amdgcn.ds.bpermute(i32 %{{[0-9]+}}, i32 %[[SRCINT]])
// CHECK:   store i32 %[[PERM]], ptr addrspace(5) %retval, align 4
// CHECK:   %[[COERCE_OUT:.*]] = getelementptr inbounds{{.*}} %union.U, ptr addrspace(5) %retval, i32 0, i32 0
// CHECK:   %[[RES:.*]] = load i32, ptr addrspace(5) %[[COERCE_OUT]], align 4
// CHECK:   ret i32 %[[RES]]
extern "C" __device__ U test_source_union(int idx, U src) {
  return __builtin_amdgcn_ds_bpermute(idx, src);
}

// CHECK-LABEL: define{{.*}}{ double, double } @test_source_cdouble(i32 {{[^,]*}}, double noundef %src.coerce0, double noundef %src.coerce1)
// Emit complex to bperm.src, load as i128, permute 4 words, store back
// CHECK:   store double %src1.real, ptr %bperm.src.ascast.realp, align 8
// CHECK:   store double %src1.imag, ptr %bperm.src.ascast.imagp, align 8
// CHECK:   %[[SRCINT:.*]] = load i128, ptr %bperm.src.ascast, align 8
// Word 0
// CHECK:   %[[W0:.*]] = trunc i128 %[[SRCINT]] to i32
// CHECK:   %[[R0:.*]] = call i32 @llvm.amdgcn.ds.bpermute(i32 %{{[0-9]+}}, i32 %[[W0]])
// CHECK:   %[[Z0:.*]] = zext i32 %[[R0]] to i128
// Word 1
// CHECK:   %[[SHR1:.*]] = lshr i128 %[[SRCINT]], 32
// CHECK:   %[[W1:.*]] = trunc i128 %[[SHR1]] to i32
// CHECK:   %[[R1:.*]] = call i32 @llvm.amdgcn.ds.bpermute(i32 %{{[0-9]+}}, i32 %[[W1]])
// CHECK:   %[[Z1:.*]] = zext i32 %[[R1]] to i128
// CHECK:   %[[S1:.*]] = shl i128 %[[Z1]], 32
// CHECK:   %[[OR1:.*]] = or i128 %[[Z0]], %[[S1]]
// Word 2
// CHECK:   %[[SHR2:.*]] = lshr i128 %[[SRCINT]], 64
// CHECK:   %[[W2:.*]] = trunc i128 %[[SHR2]] to i32
// CHECK:   %[[R2:.*]] = call i32 @llvm.amdgcn.ds.bpermute(i32 %{{[0-9]+}}, i32 %[[W2]])
// CHECK:   %[[Z2:.*]] = zext i32 %[[R2]] to i128
// CHECK:   %[[S2:.*]] = shl i128 %[[Z2]], 64
// CHECK:   %[[OR2:.*]] = or i128 %[[OR1]], %[[S2]]
// Word 3
// CHECK:   %[[SHR3:.*]] = lshr i128 %[[SRCINT]], 96
// CHECK:   %[[W3:.*]] = trunc i128 %[[SHR3]] to i32
// CHECK:   %[[R3:.*]] = call i32 @llvm.amdgcn.ds.bpermute(i32 %{{[0-9]+}}, i32 %[[W3]])
// CHECK:   %[[Z3:.*]] = zext i32 %[[R3]] to i128
// CHECK:   %[[S3:.*]] = shl i128 %[[Z3]], 96
// CHECK:   %[[OR3:.*]] = or i128 %[[OR2]], %[[S3]]
// Store result integer to agg.tmp, reconstruct complex, return
// CHECK:   store i128 %[[OR3]], ptr %agg.tmp.ascast, align 8
// CHECK:   %[[AGG_REAL:.*]] = load double, ptr %agg.tmp.ascast.realp, align 8
// CHECK:   %[[AGG_IMAG:.*]] = load double, ptr %agg.tmp.ascast.imagp, align 8
// CHECK:   store double %[[AGG_REAL]], ptr addrspace(5) %retval.realp, align 8
// CHECK:   store double %[[AGG_IMAG]], ptr addrspace(5) %retval.imagp, align 8
// CHECK:   %[[RETVAL:.*]] = load { double, double }, ptr addrspace(5) %retval, align 8
// CHECK:   ret { double, double } %[[RETVAL]]
extern "C" __device__ _Complex double test_source_cdouble(int idx, _Complex double src) {
  return __builtin_amdgcn_ds_bpermute(idx, src);
}

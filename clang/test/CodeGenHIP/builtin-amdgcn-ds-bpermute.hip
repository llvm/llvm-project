// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -target-cpu gfx1200 -aux-triple x86_64-pc-linux-gnu \
// RUN:    -fcuda-is-device -emit-llvm -o - %s | FileCheck %s

typedef short short2 __attribute__((vector_size(4)));
typedef double double2 __attribute__((ext_vector_type(2)));

struct Inner { short a; char b; };
struct Outer { int x; struct Inner y; char z; };

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
// CHECK: [[BC:%.*]] = bitcast i64 {{.*}} to <2 x i32>
// CHECK: [[LO:%.*]] = extractelement <2 x i32> [[BC]], i32 0
// CHECK: [[RLO:%.*]] = call i32 @llvm.amdgcn.ds.bpermute(i32 %0, i32 [[LO]])
// CHECK: [[V0:%.*]] = insertelement <2 x i32> poison, i32 [[RLO]], i32 0
// CHECK: [[HI:%.*]] = extractelement <2 x i32> [[BC]], i32 1
// CHECK: [[RHI:%.*]] = call i32 @llvm.amdgcn.ds.bpermute(i32 %0, i32 [[HI]])
// CHECK: [[V1:%.*]] = insertelement <2 x i32> [[V0]], i32 [[RHI]], i32 1
// CHECK: [[RES:%.*]] = bitcast <2 x i32> [[V1]] to i64
// CHECK: ret i64 [[RES]]
extern "C" __device__ long test_source_long(int a, long c) {
   return __builtin_amdgcn_ds_bpermute(a, c);
}

// CHECK-LABEL: define{{.*}}@test_source_ulong
// CHECK: [[BC:%.*]] = bitcast i64 {{.*}} to <2 x i32>
// CHECK: [[LO:%.*]] = extractelement <2 x i32> [[BC]], i32 0
// CHECK: [[RLO:%.*]] = call i32 @llvm.amdgcn.ds.bpermute(i32 %0, i32 [[LO]])
// CHECK: [[V0:%.*]] = insertelement <2 x i32> poison, i32 [[RLO]], i32 0
// CHECK: [[HI:%.*]] = extractelement <2 x i32> [[BC]], i32 1
// CHECK: [[RHI:%.*]] = call i32 @llvm.amdgcn.ds.bpermute(i32 %0, i32 [[HI]])
// CHECK: [[V1:%.*]] = insertelement <2 x i32> [[V0]], i32 [[RHI]], i32 1
// CHECK: [[RES:%.*]] = bitcast <2 x i32> [[V1]] to i64
// CHECK: ret i64 [[RES]]
extern "C" __device__ unsigned long test_source_ulong(int a, unsigned long c) {
   return __builtin_amdgcn_ds_bpermute(a, c);
}

// CHECK-LABEL: define{{.*}}@test_source_double
// CHECK: [[BC:%.*]] = bitcast double {{.*}} to <2 x i32>
// CHECK: [[LO:%.*]] = extractelement <2 x i32> [[BC]], i32 0
// CHECK: [[RLO:%.*]] = call i32 @llvm.amdgcn.ds.bpermute(i32 %0, i32 [[LO]])
// CHECK: [[V0:%.*]] = insertelement <2 x i32> poison, i32 [[RLO]], i32 0
// CHECK: [[HI:%.*]] = extractelement <2 x i32> [[BC]], i32 1
// CHECK: [[RHI:%.*]] = call i32 @llvm.amdgcn.ds.bpermute(i32 %0, i32 [[HI]])
// CHECK: [[V1:%.*]] = insertelement <2 x i32> [[V0]], i32 [[RHI]], i32 1
// CHECK: [[RES:%.*]] = bitcast <2 x i32> [[V1]] to double
// CHECK: ret double [[RES]]
extern "C" __device__ double test_source_double(int a, double c) {
   return __builtin_amdgcn_ds_bpermute(a, c);
}

// CHECK-LABEL: define{{.*}}@test_source_ptr
// CHECK: [[P2I:%.*]] = ptrtoint ptr {{.*}} to i64
// CHECK: [[BC:%.*]] = bitcast i64 [[P2I]] to <2 x i32>
// CHECK: [[LO:%.*]] = extractelement <2 x i32> [[BC]], i32 0
// CHECK: [[RLO:%.*]] = call i32 @llvm.amdgcn.ds.bpermute(i32 %0, i32 [[LO]])
// CHECK: [[V0:%.*]] = insertelement <2 x i32> poison, i32 [[RLO]], i32 0
// CHECK: [[HI:%.*]] = extractelement <2 x i32> [[BC]], i32 1
// CHECK: [[RHI:%.*]] = call i32 @llvm.amdgcn.ds.bpermute(i32 %0, i32 [[HI]])
// CHECK: [[V1:%.*]] = insertelement <2 x i32> [[V0]], i32 [[RHI]], i32 1
// CHECK: [[I64RES:%.*]] = bitcast <2 x i32> [[V1]] to i64
// CHECK: [[PRES:%.*]] = inttoptr i64 [[I64RES]] to ptr
// CHECK: ret ptr [[PRES]]
extern "C" __device__ void* test_source_ptr(int a, void* c) {
  return __builtin_amdgcn_ds_bpermute(a, c);
}

// CHECK-LABEL: define{{.*}}@test_source_double2
// CHECK: [[BC:%.*]] = bitcast <2 x double> {{.*}} to <4 x i32>
// CHECK: [[E0:%.*]] = extractelement <4 x i32> [[BC]], i32 0
// CHECK: [[R0:%.*]] = call i32 @llvm.amdgcn.ds.bpermute(i32 %0, i32 [[E0]])
// CHECK: [[V0:%.*]] = insertelement <4 x i32> poison, i32 [[R0]], i32 0
// CHECK: [[E1:%.*]] = extractelement <4 x i32> [[BC]], i32 1
// CHECK: [[R1:%.*]] = call i32 @llvm.amdgcn.ds.bpermute(i32 %0, i32 [[E1]])
// CHECK: [[V1:%.*]] = insertelement <4 x i32> [[V0]], i32 [[R1]], i32 1
// CHECK: [[E2:%.*]] = extractelement <4 x i32> [[BC]], i32 2
// CHECK: [[R2:%.*]] = call i32 @llvm.amdgcn.ds.bpermute(i32 %0, i32 [[E2]])
// CHECK: [[V2:%.*]] = insertelement <4 x i32> [[V1]], i32 [[R2]], i32 2
// CHECK: [[E3:%.*]] = extractelement <4 x i32> [[BC]], i32 3
// CHECK: [[R3:%.*]] = call i32 @llvm.amdgcn.ds.bpermute(i32 %0, i32 [[E3]])
// CHECK: [[V3:%.*]] = insertelement <4 x i32> [[V2]], i32 [[R3]], i32 3
// CHECK: [[RES:%.*]] = bitcast <4 x i32> [[V3]] to <2 x double>
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
// CHECK: entry:
// CHECK:   %retval = alloca %struct.Outer, align 4, addrspace(5)
// CHECK:   %src = alloca %struct.Outer, align 4, addrspace(5)
// CHECK:   %idx.addr = alloca i32, align 4, addrspace(5)
// CHECK:   %dsbperm.src = alloca %struct.Outer, align 4, addrspace(5)
// CHECK:   %retval.ascast = addrspacecast ptr addrspace(5) %retval to ptr
// CHECK:   %src1 = addrspacecast ptr addrspace(5) %src to ptr
// CHECK:   %idx.addr.ascast = addrspacecast ptr addrspace(5) %idx.addr to ptr
// CHECK:   %dsbperm.src.ascast = addrspacecast ptr addrspace(5) %dsbperm.src to ptr
// Materialize src aggregate from coerced pieces
// CHECK:   %[[SRC0_GEP:.*]] = getelementptr inbounds nuw %struct.Outer, ptr %src1, i32 0, i32 0
// CHECK:   store i32 %src.coerce0, ptr %[[SRC0_GEP]], align 4
// CHECK:   %[[SRC1_GEP:.*]] = getelementptr inbounds nuw %struct.Outer, ptr %src1, i32 0, i32 1
// CHECK:   store %struct.Inner %src.coerce1, ptr %[[SRC1_GEP]], align 4
// CHECK:   %[[SRC2_GEP:.*]] = getelementptr inbounds nuw %struct.Outer, ptr %src1, i32 0, i32 2
// CHECK:   store i8 %src.coerce2, ptr %[[SRC2_GEP]], align 4
// CHECK:   store i32 %idx, ptr %idx.addr.ascast, align 4
// CHECK:   %[[IDX:.*]] = load i32, ptr %idx.addr.ascast, align 4
// Forward src bytes to source buffer
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %dsbperm.src.ascast, ptr align 4 %src1, i64 12, i1 false)

// First 4-byte word at offset 0 -> write directly to retval
// CHECK:   %[[SRC0:.*]] = getelementptr inbounds i8, ptr %dsbperm.src.ascast, i64 0
// CHECK:   %[[DST0:.*]] = getelementptr inbounds i8, ptr %retval.ascast, i64 0
// CHECK:   %[[LD0:.*]] = load i32, ptr %[[SRC0]], align 4
// CHECK:   %[[P0:.*]] = call i32 @llvm.amdgcn.ds.bpermute(i32 %[[IDX]], i32 %[[LD0]])
// CHECK:   store i32 %[[P0]], ptr %[[DST0]], align 4

// Second 4-byte word at offset 4
// CHECK:   %[[SRC1B:.*]] = getelementptr inbounds i8, ptr %dsbperm.src.ascast, i64 4
// CHECK:   %[[DST1:.*]] = getelementptr inbounds i8, ptr %retval.ascast, i64 4
// CHECK:   %[[LD1:.*]] = load i32, ptr %[[SRC1B]], align 4
// CHECK:   %[[P1:.*]] = call i32 @llvm.amdgcn.ds.bpermute(i32 %[[IDX]], i32 %[[LD1]])
// CHECK:   store i32 %[[P1]], ptr %[[DST1]], align 4

// Third 4-byte word at offset 8 (size is 12 bytes total)
// CHECK:   %[[SRC2B:.*]] = getelementptr inbounds i8, ptr %dsbperm.src.ascast, i64 8
// CHECK:   %[[DST2:.*]] = getelementptr inbounds i8, ptr %retval.ascast, i64 8
// CHECK:   %[[LD2:.*]] = load i32, ptr %[[SRC2B]], align 4
// CHECK:   %[[P2:.*]] = call i32 @llvm.amdgcn.ds.bpermute(i32 %[[IDX]], i32 %[[LD2]])
// CHECK:   store i32 %[[P2]], ptr %[[DST2]], align 4

// Return the aggregate from retval.ascast
// CHECK:   %[[RES:.*]] = load %struct.Outer, ptr %retval.ascast, align 4
// CHECK:   ret %struct.Outer %[[RES]]
extern "C" __device__ Outer test_source_nested(int idx, Outer src) {
  return __builtin_amdgcn_ds_bpermute(idx, src);
}

// CHECK-LABEL: define{{.*}}@test_source_union(
// CHECK: entry:
// CHECK:   %retval = alloca %union.U, align 4, addrspace(5)
// CHECK:   %src = alloca %union.U, align 4, addrspace(5)
// CHECK:   %idx.addr = alloca i32, align 4, addrspace(5)
// CHECK:   %dsbperm.src = alloca %union.U, align 4, addrspace(5)
// CHECK:   %retval.ascast = addrspacecast ptr addrspace(5) %retval to ptr
// CHECK:   %src1 = addrspacecast ptr addrspace(5) %src to ptr
// CHECK:   %idx.addr.ascast = addrspacecast ptr addrspace(5) %idx.addr to ptr
// CHECK:   %dsbperm.src.ascast = addrspacecast ptr addrspace(5) %dsbperm.src to ptr
// Materialize src union from coerced piece
// CHECK:   %[[COERCE_DST:.*]] = getelementptr inbounds nuw %union.U, ptr %src1, i32 0, i32 0
// CHECK:   store i32 %src.coerce, ptr %[[COERCE_DST]], align 4
// CHECK:   store i32 %idx, ptr %idx.addr.ascast, align 4
// CHECK:   %[[IDX:.*]] = load i32, ptr %idx.addr.ascast, align 4
// Forward src bytes to source buffer
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %dsbperm.src.ascast, ptr align 4 %src1, i64 4, i1 false)
// Single 4-byte word -> write directly to retval
// CHECK:   %[[SRC0:.*]] = getelementptr inbounds i8, ptr %dsbperm.src.ascast, i64 0
// CHECK:   %[[DST0:.*]] = getelementptr inbounds i8, ptr %retval.ascast, i64 0
// CHECK:   %[[LD0:.*]] = load i32, ptr %[[SRC0]], align 4
// CHECK:   %[[P0:.*]] = call i32 @llvm.amdgcn.ds.bpermute(i32 %[[IDX]], i32 %[[LD0]])
// CHECK:   store i32 %[[P0]], ptr %[[DST0]], align 4
// Coerce return from retval
// CHECK:   %[[COERCE_OUT:.*]] = getelementptr inbounds{{.*}} %union.U, ptr %retval.ascast, i32 0, i32 0
// CHECK:   %[[RES:.*]] = load i32, ptr %[[COERCE_OUT]], align 4
// CHECK:   ret i32 %[[RES]]
extern "C" __device__ U test_source_union(int idx, U src) {
  return __builtin_amdgcn_ds_bpermute(idx, src);
}

// CHECK-LABEL: define{{.*}}{ double, double } @test_source_cdouble(i32 {{[^,]*}}, double noundef %src.coerce0, double noundef %src.coerce1)
// Materialize the coerced _Complex double argument into the local aggregate
// CHECK:   %retval = alloca { double, double }, align 8, addrspace(5)
// CHECK:   %src = alloca { double, double }, align 8, addrspace(5)
// CHECK:   %idx.addr = alloca i32, align 4, addrspace(5)
// CHECK:   %agg.tmp = alloca { double, double }, align 8, addrspace(5)
// CHECK:   %dsbperm.src = alloca { double, double }, align 8, addrspace(5)
// CHECK:   %retval.ascast = addrspacecast ptr addrspace(5) %retval to ptr
// CHECK:   %src1 = addrspacecast ptr addrspace(5) %src to ptr
// CHECK:   %idx.addr.ascast = addrspacecast ptr addrspace(5) %idx.addr to ptr
// CHECK:   %agg.tmp.ascast = addrspacecast ptr addrspace(5) %agg.tmp to ptr
// CHECK:   %dsbperm.src.ascast = addrspacecast ptr addrspace(5) %dsbperm.src to ptr
// CHECK:   %[[SRC0_GEP:.*]] = getelementptr inbounds nuw { double, double }, ptr %src1, i32 0, i32 0
// CHECK:   store double %src.coerce0, ptr %[[SRC0_GEP]], align 8
// CHECK:   %[[SRC1_GEP:.*]] = getelementptr inbounds nuw { double, double }, ptr %src1, i32 0, i32 1
// CHECK:   store double %src.coerce1, ptr %[[SRC1_GEP]], align 8
// Load the real/imag parts and forward into the source buffer for permutation
// CHECK:   store i32 %idx, ptr %idx.addr.ascast, align 4
// CHECK:   %[[IDX:.*]] = load i32, ptr %idx.addr.ascast, align 4
// CHECK:   %src1.realp = getelementptr inbounds nuw { double, double }, ptr %src1, i32 0, i32 0
// CHECK:   %[[SRC_REAL:.*]] = load double, ptr %src1.realp, align 8
// CHECK:   %src1.imagp = getelementptr inbounds nuw { double, double }, ptr %src1, i32 0, i32 1
// CHECK:   %[[SRC_IMAG:.*]] = load double, ptr %src1.imagp, align 8
// CHECK:   %[[SRCBUF_REALP:.*]] = getelementptr inbounds nuw { double, double }, ptr %dsbperm.src.ascast, i32 0, i32 0
// CHECK:   %[[SRCBUF_IMAGP:.*]] = getelementptr inbounds nuw { double, double }, ptr %dsbperm.src.ascast, i32 0, i32 1
// CHECK:   store double %[[SRC_REAL]], ptr %[[SRCBUF_REALP]], align 8
// CHECK:   store double %[[SRC_IMAG]], ptr %[[SRCBUF_IMAGP]], align 8
// Split the complex double into 32-bit words and pass each to the intrinsic; write into agg.tmp
// CHECK:   %[[SRC_I8_0:.*]] = getelementptr inbounds i8, ptr %dsbperm.src.ascast, i64 0
// CHECK:   %[[DST_I8_0:.*]] = getelementptr inbounds i8, ptr %agg.tmp.ascast, i64 0
// CHECK:   %[[LDW0:.*]] = load i32, ptr %[[SRC_I8_0]], align 8
// CHECK:   %[[P0:.*]] = call i32 @llvm.amdgcn.ds.bpermute(i32 %[[IDX]], i32 %[[LDW0]])
// CHECK:   store i32 %[[P0]], ptr %[[DST_I8_0]], align 8
// CHECK:   %[[SRC_I8_4:.*]] = getelementptr inbounds i8, ptr %dsbperm.src.ascast, i64 4
// CHECK:   %[[DST_I8_4:.*]] = getelementptr inbounds i8, ptr %agg.tmp.ascast, i64 4
// CHECK:   %[[LDW1:.*]] = load i32, ptr %[[SRC_I8_4]], align 4
// CHECK:   %[[P1:.*]] = call i32 @llvm.amdgcn.ds.bpermute(i32 %[[IDX]], i32 %[[LDW1]])
// CHECK:   store i32 %[[P1]], ptr %[[DST_I8_4]], align 4
// CHECK:   %[[SRC_I8_8:.*]] = getelementptr inbounds i8, ptr %dsbperm.src.ascast, i64 8
// CHECK:   %[[DST_I8_8:.*]] = getelementptr inbounds i8, ptr %agg.tmp.ascast, i64 8
// CHECK:   %[[LDW2:.*]] = load i32, ptr %[[SRC_I8_8]], align 8
// CHECK:   %[[P2:.*]] = call i32 @llvm.amdgcn.ds.bpermute(i32 %[[IDX]], i32 %[[LDW2]])
// CHECK:   store i32 %[[P2]], ptr %[[DST_I8_8]], align 8
// CHECK:   %[[SRC_I8_12:.*]] = getelementptr inbounds i8, ptr %dsbperm.src.ascast, i64 12
// CHECK:   %[[DST_I8_12:.*]] = getelementptr inbounds i8, ptr %agg.tmp.ascast, i64 12
// CHECK:   %[[LDW3:.*]] = load i32, ptr %[[SRC_I8_12]], align 4
// CHECK:   %[[P3:.*]] = call i32 @llvm.amdgcn.ds.bpermute(i32 %[[IDX]], i32 %[[LDW3]])
// CHECK:   store i32 %[[P3]], ptr %[[DST_I8_12]], align 4
// Reconstruct the complex double into retval and return it
// CHECK:   %[[AGG_REALP:.*]] = getelementptr inbounds nuw { double, double }, ptr %agg.tmp.ascast, i32 0, i32 0
// CHECK:   %[[AGG_REAL:.*]] = load double, ptr %[[AGG_REALP]], align 8
// CHECK:   %[[AGG_IMAGP:.*]] = getelementptr inbounds nuw { double, double }, ptr %agg.tmp.ascast, i32 0, i32 1
// CHECK:   %[[AGG_IMAG:.*]] = load double, ptr %[[AGG_IMAGP]], align 8
// CHECK:   %[[RET_REALP:.*]] = getelementptr inbounds nuw { double, double }, ptr %retval.ascast, i32 0, i32 0
// CHECK:   %[[RET_IMAGP:.*]] = getelementptr inbounds nuw { double, double }, ptr %retval.ascast, i32 0, i32 1
// CHECK:   store double %[[AGG_REAL]], ptr %[[RET_REALP]], align 8
// CHECK:   store double %[[AGG_IMAG]], ptr %[[RET_IMAGP]], align 8
// CHECK:   %[[RETVAL_AGG:.*]] = load { double, double }, ptr %retval.ascast, align 8
// CHECK:   ret { double, double } %[[RETVAL_AGG]]
extern "C" __device__ _Complex double test_source_cdouble(int idx, _Complex double src) {
  return __builtin_amdgcn_ds_bpermute(idx, src);
}

// RUN: %clang_cc1 -verify -triple x86_64-apple-darwin10 -fopenmp -x c -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -fopenmp -x c -triple x86_64-apple-darwin10 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -x c -triple x86_64-apple-darwin10 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s
// expected-no-diagnostics
// REQUIRES: x86-registered-target
#ifndef HEADER
#define HEADER

_Bool bv, bx;
char cv, cx;
unsigned char ucv, ucx;
short sv, sx;
unsigned short usv, usx;
int iv, ix;
unsigned int uiv, uix;
long lv, lx;
unsigned long ulv, ulx;
long long llv, llx;
unsigned long long ullv, ullx;
float fv, fx;
double dv, dx;
long double ldv, ldx;
_Complex int civ, cix;
_Complex float cfv, cfx;
_Complex double cdv, cdx;

typedef int int4 __attribute__((__vector_size__(16)));
int4 int4x;

struct BitFields {
  int : 32;
  int a : 31;
} bfx;

struct BitFields_packed {
  int : 32;
  int a : 31;
} __attribute__ ((__packed__)) bfx_packed;

struct BitFields2 {
  int : 31;
  int a : 1;
} bfx2;

struct BitFields2_packed {
  int : 31;
  int a : 1;
} __attribute__ ((__packed__)) bfx2_packed;

struct BitFields3 {
  int : 11;
  int a : 14;
} bfx3;

struct BitFields3_packed {
  int : 11;
  int a : 14;
} __attribute__ ((__packed__)) bfx3_packed;

struct BitFields4 {
  short : 16;
  int a: 1;
  long b : 7;
} bfx4;

struct BitFields4_packed {
  short : 16;
  int a: 1;
  long b : 7;
} __attribute__ ((__packed__)) bfx4_packed;

typedef float float2 __attribute__((ext_vector_type(2)));
float2 float2x;

// Register "0" is currently an invalid register for global register variables.
// Use "esp" instead of "0".
// register int rix __asm__("0");
register int rix __asm__("esp");

int main() {
// CHECK: load i8, i8*
// CHECK: store atomic i8
#pragma omp atomic write
  bx = bv;
// CHECK: load i8, i8*
// CHECK: store atomic i8
#pragma omp atomic write
  cx = cv;
// CHECK: load i8, i8*
// CHECK: store atomic i8
#pragma omp atomic write
  ucx = ucv;
// CHECK: load i16, i16*
// CHECK: store atomic i16
#pragma omp atomic write
  sx = sv;
// CHECK: load i16, i16*
// CHECK: store atomic i16
#pragma omp atomic write
  usx = usv;
// CHECK: load i32, i32*
// CHECK: store atomic i32
#pragma omp atomic write
  ix = iv;
// CHECK: load i32, i32*
// CHECK: store atomic i32
#pragma omp atomic write
  uix = uiv;
// CHECK: load i64, i64*
// CHECK: store atomic i64
#pragma omp atomic write
  lx = lv;
// CHECK: load i64, i64*
// CHECK: store atomic i64
#pragma omp atomic write
  ulx = ulv;
// CHECK: load i64, i64*
// CHECK: store atomic i64
#pragma omp atomic write
  llx = llv;
// CHECK: load i64, i64*
// CHECK: store atomic i64
#pragma omp atomic write
  ullx = ullv;
// CHECK: load float, float*
// CHECK: bitcast float {{.*}} to i32
// CHECK: store atomic i32 {{.*}}, i32* bitcast (float*
#pragma omp atomic write
  fx = fv;
// CHECK: load double, double*
// CHECK: bitcast double {{.*}} to i64
// CHECK: store atomic i64 {{.*}}, i64* bitcast (double*
#pragma omp atomic write
  dx = dv;
// CHECK: [[LD:%.+]] = load x86_fp80, x86_fp80*
// CHECK: [[BITCAST:%.+]] = bitcast x86_fp80* [[LDTEMP:%.*]] to i8*
// CHECK: call void @llvm.memset.p0i8.i64(i8* [[BITCAST]], i8 0, i64 16, i32 16, i1 false)
// CHECK: store x86_fp80 [[LD]], x86_fp80* [[LDTEMP]]
// CHECK: [[BITCAST:%.+]] = bitcast x86_fp80* [[LDTEMP:%.*]] to i128*
// CHECK: [[LD:%.+]] = load i128, i128* [[BITCAST]]
// CHECK: store atomic i128 [[LD]], i128* bitcast (x86_fp80*
#pragma omp atomic write
  ldx = ldv;
// CHECK: [[REAL_VAL:%.+]] = load i32, i32* getelementptr inbounds ({ i32, i32 }, { i32, i32 }* @{{.*}}, i32 0, i32 0)
// CHECK: [[IMG_VAL:%.+]] = load i32, i32* getelementptr inbounds ({ i32, i32 }, { i32, i32 }* @{{.*}}, i32 0, i32 1)
// CHECK: [[TEMP_REAL_REF:%.+]] = getelementptr inbounds { i32, i32 }, { i32, i32 }* [[TEMP:%.+]], i32 0, i32 0
// CHECK: [[TEMP_IMG_REF:%.+]] = getelementptr inbounds { i32, i32 }, { i32, i32 }* [[TEMP]], i32 0, i32 1
// CHECK: store i32 [[REAL_VAL]], i32* [[TEMP_REAL_REF]]
// CHECK: store i32 [[IMG_VAL]], i32* [[TEMP_IMG_REF]]
// CHECK: [[BITCAST:%.+]] = bitcast { i32, i32 }* [[TEMP]] to i8*
// CHECK: call void @__atomic_store(i64 8, i8* bitcast ({ i32, i32 }* @{{.*}} to i8*), i8* [[BITCAST]], i32 0)
#pragma omp atomic write
  cix = civ;
// CHECK: [[REAL_VAL:%.+]] = load float, float* getelementptr inbounds ({ float, float }, { float, float }* @{{.*}}, i32 0, i32 0)
// CHECK: [[IMG_VAL:%.+]] = load float, float* getelementptr inbounds ({ float, float }, { float, float }* @{{.*}}, i32 0, i32 1)
// CHECK: [[TEMP_REAL_REF:%.+]] = getelementptr inbounds { float, float }, { float, float }* [[TEMP:%.+]], i32 0, i32 0
// CHECK: [[TEMP_IMG_REF:%.+]] = getelementptr inbounds { float, float }, { float, float }* [[TEMP]], i32 0, i32 1
// CHECK: store float [[REAL_VAL]], float* [[TEMP_REAL_REF]]
// CHECK: store float [[IMG_VAL]], float* [[TEMP_IMG_REF]]
// CHECK: [[BITCAST:%.+]] = bitcast { float, float }* [[TEMP]] to i8*
// CHECK: call void @__atomic_store(i64 8, i8* bitcast ({ float, float }* @{{.*}} to i8*), i8* [[BITCAST]], i32 0)
#pragma omp atomic write
  cfx = cfv;
// CHECK: [[REAL_VAL:%.+]] = load double, double* getelementptr inbounds ({ double, double }, { double, double }* @{{.*}}, i32 0, i32 0)
// CHECK: [[IMG_VAL:%.+]] = load double, double* getelementptr inbounds ({ double, double }, { double, double }* @{{.*}}, i32 0, i32 1)
// CHECK: [[TEMP_REAL_REF:%.+]] = getelementptr inbounds { double, double }, { double, double }* [[TEMP:%.+]], i32 0, i32 0
// CHECK: [[TEMP_IMG_REF:%.+]] = getelementptr inbounds { double, double }, { double, double }* [[TEMP]], i32 0, i32 1
// CHECK: store double [[REAL_VAL]], double* [[TEMP_REAL_REF]]
// CHECK: store double [[IMG_VAL]], double* [[TEMP_IMG_REF]]
// CHECK: [[BITCAST:%.+]] = bitcast { double, double }* [[TEMP]] to i8*
// CHECK: call void @__atomic_store(i64 16, i8* bitcast ({ double, double }* @{{.*}} to i8*), i8* [[BITCAST]], i32 5)
// CHECK: call{{.*}} @__kmpc_flush(
#pragma omp atomic seq_cst write
  cdx = cdv;
// CHECK: load i8, i8*
// CHECK: store atomic i64
#pragma omp atomic write
  ulx = bv;
// CHECK: load i8, i8*
// CHECK: store atomic i8
#pragma omp atomic write
  bx = cv;
// CHECK: load i8, i8*
// CHECK: store atomic i8
// CHECK: call{{.*}} @__kmpc_flush(
#pragma omp atomic write, seq_cst
  cx = ucv;
// CHECK: load i16, i16*
// CHECK: store atomic i64
#pragma omp atomic write
  ulx = sv;
// CHECK: load i16, i16*
// CHECK: store atomic i64
#pragma omp atomic write
  lx = usv;
// CHECK: load i32, i32*
// CHECK: store atomic i32
// CHECK: call{{.*}} @__kmpc_flush(
#pragma omp atomic seq_cst, write
  uix = iv;
// CHECK: load i32, i32*
// CHECK: store atomic i32
#pragma omp atomic write
  ix = uiv;
// CHECK: load i64, i64*
// CHECK: [[VAL:%.+]] = trunc i64 %{{.*}} to i32
// CHECK: [[TEMP_REAL_REF:%.+]] = getelementptr inbounds { i32, i32 }, { i32, i32 }* [[TEMP:%.+]], i32 0, i32 0
// CHECK: [[TEMP_IMG_REF:%.+]] = getelementptr inbounds { i32, i32 }, { i32, i32 }* [[TEMP]], i32 0, i32 1
// CHECK: store i32 [[VAL]], i32* [[TEMP_REAL_REF]]
// CHECK: store i32 0, i32* [[TEMP_IMG_REF]]
// CHECK: [[BITCAST:%.+]] = bitcast { i32, i32 }* [[TEMP]] to i8*
// CHECK: call void @__atomic_store(i64 8, i8* bitcast ({ i32, i32 }* @{{.+}} to i8*), i8* [[BITCAST]], i32 0)
#pragma omp atomic write
  cix = lv;
// CHECK: load i64, i64*
// CHECK: store atomic i32 %{{.+}}, i32* bitcast (float*
#pragma omp atomic write
  fx = ulv;
// CHECK: load i64, i64*
// CHECK: store atomic i64 %{{.+}}, i64* bitcast (double*
#pragma omp atomic write
  dx = llv;
// CHECK: load i64, i64*
// CHECK: [[VAL:%.+]] = uitofp i64 %{{.+}} to x86_fp80
// CHECK: [[BITCAST:%.+]] = bitcast x86_fp80* [[TEMP:%.+]] to i8*
// CHECK: call void @llvm.memset.p0i8.i64(i8* [[BITCAST]], i8 0, i64 16, i32 16, i1 false)
// CHECK: store x86_fp80 [[VAL]], x86_fp80* [[TEMP]]
// CHECK: [[BITCAST:%.+]] = bitcast x86_fp80* [[TEMP]] to i128*
// CHECK: [[VAL:%.+]] = load i128, i128* [[BITCAST]]
// CHECK: store atomic i128 [[VAL]], i128* bitcast (x86_fp80*
#pragma omp atomic write
  ldx = ullv;
// CHECK: load float, float*
// CHECK: [[VAL:%.+]] = fptosi float %{{.*}} to i32
// CHECK: [[TEMP_REAL_REF:%.+]] = getelementptr inbounds { i32, i32 }, { i32, i32 }* [[TEMP:%.+]], i32 0, i32 0
// CHECK: [[TEMP_IMG_REF:%.+]] = getelementptr inbounds { i32, i32 }, { i32, i32 }* [[TEMP]], i32 0, i32 1
// CHECK: store i32 [[VAL]], i32* [[TEMP_REAL_REF]]
// CHECK: store i32 0, i32* [[TEMP_IMG_REF]]
// CHECK: [[BITCAST:%.+]] = bitcast { i32, i32 }* [[TEMP]] to i8*
// CHECK: call void @__atomic_store(i64 8, i8* bitcast ({ i32, i32 }* @{{.+}} to i8*), i8* [[BITCAST]], i32 0)
#pragma omp atomic write
  cix = fv;
// CHECK: load double, double*
// CHECK: store atomic i16
#pragma omp atomic write
  sx = dv;
// CHECK: load x86_fp80, x86_fp80*
// CHECK: store atomic i8
#pragma omp atomic write
  bx = ldv;
// CHECK: load i32, i32* getelementptr inbounds ({ i32, i32 }, { i32, i32 }* @{{.+}}, i32 0, i32 0)
// CHECK: load i32, i32* getelementptr inbounds ({ i32, i32 }, { i32, i32 }* @{{.+}}, i32 0, i32 1)
// CHECK: icmp ne i32 %{{.+}}, 0
// CHECK: icmp ne i32 %{{.+}}, 0
// CHECK: or i1
// CHECK: store atomic i8
#pragma omp atomic write
  bx = civ;
// CHECK: load float, float* getelementptr inbounds ({ float, float }, { float, float }* @{{.*}}, i32 0, i32 0)
// CHECK: store atomic i16
#pragma omp atomic write
  usx = cfv;
// CHECK: load double, double* getelementptr inbounds ({ double, double }, { double, double }* @{{.+}}, i32 0, i32 0)
// CHECK: store atomic i64
#pragma omp atomic write
  llx = cdv;
// CHECK-DAG: [[IDX:%.+]] = load i16, i16* @{{.+}}
// CHECK-DAG: load i8, i8*
// CHECK-DAG: [[VEC_ITEM_VAL:%.+]] = zext i1 %{{.+}} to i32
// CHECK: [[I128VAL:%.+]] = load atomic i128, i128* bitcast (<4 x i32>* [[DEST:@.+]] to i128*) monotonic
// CHECK: br label %[[CONT:.+]]
// CHECK: [[CONT]]
// CHECK: [[OLD_I128:%.+]] = phi i128 [ [[I128VAL]], %{{.+}} ], [ [[FAILED_I128_OLD_VAL:%.+]], %[[CONT]] ]
// CHECK: [[BITCAST:%.+]] = bitcast <4 x i32>* [[LDTEMP:%.+]] to i128*
// CHECK: store i128 [[OLD_I128]], i128* [[BITCAST]],
// CHECK: [[VEC_VAL:%.+]] = load <4 x i32>, <4 x i32>* [[LDTEMP]]
// CHECK: [[NEW_VEC_VAL:%.+]] = insertelement <4 x i32> [[VEC_VAL]], i32 [[VEC_ITEM_VAL]], i16 [[IDX]]
// CHECK: store <4 x i32> [[NEW_VEC_VAL]], <4 x i32>* [[LDTEMP]]
// CHECK: [[NEW_I128:%.+]] = load i128, i128* [[BITCAST]]
// CHECK: [[RES:%.+]] = cmpxchg i128* bitcast (<4 x i32>* [[DEST]] to i128*), i128 [[OLD_I128]], i128 [[NEW_I128]] monotonic monotonic
// CHECK: [[FAILED_I128_OLD_VAL:%.+]] = extractvalue { i128, i1 } [[RES]], 0
// CHECK: [[FAIL_SUCCESS:%.+]] = extractvalue { i128, i1 } [[RES]], 1
// CHECK: br i1 [[FAIL_SUCCESS]], label %[[EXIT:.+]], label %[[CONT]]
// CHECK: [[EXIT]]
#pragma omp atomic write
  int4x[sv] = bv;
// CHECK: load x86_fp80, x86_fp80* @{{.+}}
// CHECK: [[NEW_VAL:%.+]] = fptosi x86_fp80 %{{.+}} to i32
// CHECK: [[PREV_VALUE:%.+]] = load atomic i32, i32* bitcast (i8* getelementptr (i8, i8* bitcast (%struct.BitFields* @{{.+}} to i8*), i64 4) to i32*) monotonic
// CHECK: br label %[[CONT:.+]]
// CHECK: [[CONT]]
// CHECK: [[OLD_BF_VALUE:%.+]] = phi i32 [ [[PREV_VALUE]], %[[EXIT]] ], [ [[FAILED_OLD_VAL:%.+]], %[[CONT]] ]
// CHECK: [[BF_VALUE:%.+]] = and i32 [[NEW_VAL]], 2147483647
// CHECK: [[BF_CLEAR:%.+]] = and i32 %{{.+}}, -2147483648
// CHECK: or i32 [[BF_CLEAR]], [[BF_VALUE]]
// CHECK: store i32 %{{.+}}, i32* [[LDTEMP:%.+]]
// CHECK: [[NEW_BF_VALUE:%.+]] = load i32, i32* [[LDTEMP]]
// CHECK: [[RES:%.+]] = cmpxchg i32* bitcast (i8* getelementptr (i8, i8* bitcast (%struct.BitFields* @{{.+}} to i8*), i64 4) to i32*), i32 [[OLD_BF_VALUE]], i32 [[NEW_BF_VALUE]] monotonic monotonic
// CHECK: [[FAILED_OLD_VAL]] = extractvalue { i32, i1 } [[RES]], 0
// CHECK: [[FAIL_SUCCESS:%.+]] = extractvalue { i32, i1 } [[RES]], 1
// CHECK: br i1 [[FAIL_SUCCESS]], label %[[EXIT:.+]], label %[[CONT]]
// CHECK: [[EXIT]]
#pragma omp atomic write
  bfx.a = ldv;
// CHECK: load x86_fp80, x86_fp80* @{{.+}}
// CHECK: [[NEW_VAL:%.+]] = fptosi x86_fp80 %{{.+}} to i32
// CHECK: [[BITCAST:%.+]] = bitcast i32* [[LDTEMP:%.+]] to i8*
// CHECK: call void @__atomic_load(i64 4, i8* getelementptr (i8, i8* bitcast (%struct.BitFields_packed* @{{.+}} to i8*), i64 4), i8* [[BITCAST]], i32 0)
// CHECK: br label %[[CONT:.+]]
// CHECK: [[CONT]]
// CHECK: [[OLD_BF_VALUE:%.+]] = load i32, i32* [[LDTEMP]],
// CHECK: store i32 [[OLD_BF_VALUE]], i32* [[LDTEMP1:%.+]],
// CHECK: [[OLD_BF_VALUE:%.+]] = load i32, i32* [[LDTEMP1]],
// CHECK: [[BF_VALUE:%.+]] = and i32 [[NEW_VAL]], 2147483647
// CHECK: [[BF_CLEAR:%.+]] = and i32 [[OLD_BF_VALUE]], -2147483648
// CHECK: or i32 [[BF_CLEAR]], [[BF_VALUE]]
// CHECK: store i32 %{{.+}}, i32* [[LDTEMP1]]
// CHECK: [[BITCAST_TEMP_OLD_BF_ADDR:%.+]] = bitcast i32* [[LDTEMP]] to i8*
// CHECK: [[BITCAST_TEMP_NEW_BF_ADDR:%.+]] = bitcast i32* [[LDTEMP1]] to i8*
// CHECK: [[FAIL_SUCCESS:%.+]] = call zeroext i1 @__atomic_compare_exchange(i64 4, i8* getelementptr (i8, i8* bitcast (%struct.BitFields_packed* @{{.+}} to i8*), i64 4), i8* [[BITCAST_TEMP_OLD_BF_ADDR]], i8* [[BITCAST_TEMP_NEW_BF_ADDR]], i32 0, i32 0)
// CHECK: br i1 [[FAIL_SUCCESS]], label %[[EXIT:.+]], label %[[CONT]]
// CHECK: [[EXIT]]
#pragma omp atomic write
  bfx_packed.a = ldv;
// CHECK: load x86_fp80, x86_fp80* @{{.+}}
// CHECK: [[NEW_VAL:%.+]] = fptosi x86_fp80 %{{.+}} to i32
// CHECK: [[PREV_VALUE:%.+]] = load atomic i32, i32* getelementptr inbounds (%struct.BitFields2, %struct.BitFields2* @{{.+}}, i32 0, i32 0) monotonic
// CHECK: br label %[[CONT:.+]]
// CHECK: [[CONT]]
// CHECK: [[OLD_BF_VALUE:%.+]] = phi i32 [ [[PREV_VALUE]], %[[EXIT]] ], [ [[FAILED_OLD_VAL:%.+]], %[[CONT]] ]
// CHECK: [[BF_AND:%.+]] = and i32 [[NEW_VAL]], 1
// CHECK: [[BF_VALUE:%.+]] = shl i32 [[BF_AND]], 31
// CHECK: [[BF_CLEAR:%.+]] = and i32 %{{.+}}, 2147483647
// CHECK: or i32 [[BF_CLEAR]], [[BF_VALUE]]
// CHECK: store i32 %{{.+}}, i32* [[LDTEMP:%.+]]
// CHECK: [[NEW_BF_VALUE:%.+]] = load i32, i32* [[LDTEMP]]
// CHECK: [[RES:%.+]] = cmpxchg i32* getelementptr inbounds (%struct.BitFields2, %struct.BitFields2* @{{.+}}, i32 0, i32 0), i32 [[OLD_BF_VALUE]], i32 [[NEW_BF_VALUE]] monotonic monotonic
// CHECK: [[FAILED_OLD_VAL]] = extractvalue { i32, i1 } [[RES]], 0
// CHECK: [[FAIL_SUCCESS:%.+]] = extractvalue { i32, i1 } [[RES]], 1
// CHECK: br i1 [[FAIL_SUCCESS]], label %[[EXIT:.+]], label %[[CONT]]
// CHECK: [[EXIT]]
#pragma omp atomic write
  bfx2.a = ldv;
// CHECK: load x86_fp80, x86_fp80* @{{.+}}
// CHECK: [[NEW_VAL:%.+]] = fptosi x86_fp80 %{{.+}} to i32
// CHECK: [[PREV_VALUE:%.+]] = load atomic i8, i8* getelementptr (i8, i8* bitcast (%struct.BitFields2_packed* @{{.+}} to i8*), i64 3) monotonic
// CHECK: br label %[[CONT:.+]]
// CHECK: [[CONT]]
// CHECK: [[OLD_BF_VALUE:%.+]] = phi i8 [ [[PREV_VALUE]], %[[EXIT]] ], [ [[FAILED_OLD_VAL:%.+]], %[[CONT]] ]
// CHECK: [[TRUNC:%.+]] = trunc i32 [[NEW_VAL]] to i8
// CHECK: [[BF_AND:%.+]] = and i8 [[TRUNC]], 1
// CHECK: [[BF_VALUE:%.+]] = shl i8 [[BF_AND]], 7
// CHECK: [[BF_CLEAR:%.+]] = and i8 %{{.+}}, 127
// CHECK: or i8 [[BF_CLEAR]], [[BF_VALUE]]
// CHECK: store i8 %{{.+}}, i8* [[LDTEMP:%.+]]
// CHECK: [[NEW_BF_VALUE:%.+]] = load i8, i8* [[LDTEMP]]
// CHECK: [[RES:%.+]] = cmpxchg i8* getelementptr (i8, i8* bitcast (%struct.BitFields2_packed* @{{.+}} to i8*), i64 3), i8 [[OLD_BF_VALUE]], i8 [[NEW_BF_VALUE]] monotonic monotonic
// CHECK: [[FAILED_OLD_VAL]] = extractvalue { i8, i1 } [[RES]], 0
// CHECK: [[FAIL_SUCCESS:%.+]] = extractvalue { i8, i1 } [[RES]], 1
// CHECK: br i1 [[FAIL_SUCCESS]], label %[[EXIT:.+]], label %[[CONT]]
// CHECK: [[EXIT]]
#pragma omp atomic write
  bfx2_packed.a = ldv;
// CHECK: load x86_fp80, x86_fp80* @{{.+}}
// CHECK: [[NEW_VAL:%.+]] = fptosi x86_fp80 %{{.+}} to i32
// CHECK: [[PREV_VALUE:%.+]] = load atomic i32, i32* getelementptr inbounds (%struct.BitFields3, %struct.BitFields3* @{{.+}}, i32 0, i32 0) monotonic
// CHECK: br label %[[CONT:.+]]
// CHECK: [[CONT]]
// CHECK: [[OLD_BF_VALUE:%.+]] = phi i32 [ [[PREV_VALUE]], %[[EXIT]] ], [ [[FAILED_OLD_VAL:%.+]], %[[CONT]] ]
// CHECK: [[BF_AND:%.+]] = and i32 [[NEW_VAL]], 16383
// CHECK: [[BF_VALUE:%.+]] = shl i32 [[BF_AND]], 11
// CHECK: [[BF_CLEAR:%.+]] = and i32 %{{.+}}, -33552385
// CHECK: or i32 [[BF_CLEAR]], [[BF_VALUE]]
// CHECK: store i32 %{{.+}}, i32* [[LDTEMP:%.+]]
// CHECK: [[NEW_BF_VALUE:%.+]] = load i32, i32* [[LDTEMP]]
// CHECK: [[RES:%.+]] = cmpxchg i32* getelementptr inbounds (%struct.BitFields3, %struct.BitFields3* @{{.+}}, i32 0, i32 0), i32 [[OLD_BF_VALUE]], i32 [[NEW_BF_VALUE]] monotonic monotonic
// CHECK: [[FAILED_OLD_VAL]] = extractvalue { i32, i1 } [[RES]], 0
// CHECK: [[FAIL_SUCCESS:%.+]] = extractvalue { i32, i1 } [[RES]], 1
// CHECK: br i1 [[FAIL_SUCCESS]], label %[[EXIT:.+]], label %[[CONT]]
// CHECK: [[EXIT]]
#pragma omp atomic write
  bfx3.a = ldv;
// CHECK: load x86_fp80, x86_fp80* @{{.+}}
// CHECK: [[NEW_VAL:%.+]] = fptosi x86_fp80 %{{.+}} to i32
// CHECK: [[LDTEMP:%.+]] = bitcast i32* %{{.+}} to i24*
// CHECK: [[BITCAST:%.+]] = bitcast i24* %{{.+}} to i8*
// CHECK: call void @__atomic_load(i64 3, i8* getelementptr (i8, i8* bitcast (%struct.BitFields3_packed* @{{.+}} to i8*), i64 1), i8* [[BITCAST]], i32 0)
// CHECK: br label %[[CONT:.+]]
// CHECK: [[CONT]]
// CHECK: [[OLD_VAL:%.+]] = load i24, i24* %{{.+}},
// CHECK: store i24 [[OLD_VAL]], i24* [[TEMP:%.+]],
// CHECK: [[TRUNC:%.+]] = trunc i32 [[NEW_VAL]] to i24
// CHECK: [[BF_AND:%.+]] = and i24 [[TRUNC]], 16383
// CHECK: [[BF_VALUE:%.+]] = shl i24 [[BF_AND]], 3
// CHECK: [[BF_CLEAR:%.+]] = and i24 %{{.+}}, -131065
// CHECK: or i24 [[BF_CLEAR]], [[BF_VALUE]]
// CHECK: store i24 %{{.+}}, i24* [[TEMP]]
// CHECK: [[BITCAST_TEMP_OLD_BF_ADDR:%.+]] = bitcast i24* [[LDTEMP]] to i8*
// CHECK: [[BITCAST_TEMP_NEW_BF_ADDR:%.+]] = bitcast i24* [[TEMP]] to i8*
// CHECK: [[FAIL_SUCCESS:%.+]] = call zeroext i1 @__atomic_compare_exchange(i64 3, i8* getelementptr (i8, i8* bitcast (%struct.BitFields3_packed* @{{.+}} to i8*), i64 1), i8* [[BITCAST_TEMP_OLD_BF_ADDR]], i8* [[BITCAST_TEMP_NEW_BF_ADDR]], i32 0, i32 0)
// CHECK: br i1 [[FAIL_SUCCESS]], label %[[EXIT:.+]], label %[[CONT]]
// CHECK: [[EXIT]]
#pragma omp atomic write
  bfx3_packed.a = ldv;
// CHECK: load x86_fp80, x86_fp80* @{{.+}}
// CHECK: [[NEW_VAL:%.+]] = fptosi x86_fp80 %{{.+}} to i32
// CHECK: [[PREV_VALUE:%.+]] = load atomic i64, i64* bitcast (%struct.BitFields4* @{{.+}} to i64*) monotonic
// CHECK: br label %[[CONT:.+]]
// CHECK: [[CONT]]
// CHECK: [[OLD_BF_VALUE:%.+]] = phi i64 [ [[PREV_VALUE]], %[[EXIT]] ], [ [[FAILED_OLD_VAL:%.+]], %[[CONT]] ]
// CHECK: [[ZEXT:%.+]] = zext i32 [[NEW_VAL]] to i64
// CHECK: [[BF_AND:%.+]] = and i64 [[ZEXT]], 1
// CHECK: [[BF_VALUE:%.+]] = shl i64 [[BF_AND]], 16
// CHECK: [[BF_CLEAR:%.+]] = and i64 %{{.+}}, -65537
// CHECK: or i64 [[BF_CLEAR]], [[BF_VALUE]]
// CHECK: store i64 %{{.+}}, i64* [[LDTEMP:%.+]]
// CHECK: [[NEW_BF_VALUE:%.+]] = load i64, i64* [[LDTEMP]]
// CHECK: [[RES:%.+]] = cmpxchg i64* bitcast (%struct.BitFields4* @{{.+}} to i64*), i64 [[OLD_BF_VALUE]], i64 [[NEW_BF_VALUE]] monotonic monotonic
// CHECK: [[FAILED_OLD_VAL]] = extractvalue { i64, i1 } [[RES]], 0
// CHECK: [[FAIL_SUCCESS:%.+]] = extractvalue { i64, i1 } [[RES]], 1
// CHECK: br i1 [[FAIL_SUCCESS]], label %[[EXIT:.+]], label %[[CONT]]
// CHECK: [[EXIT]]
#pragma omp atomic write
  bfx4.a = ldv;
// CHECK: load x86_fp80, x86_fp80* @{{.+}}
// CHECK: [[NEW_VAL:%.+]] = fptosi x86_fp80 %{{.+}} to i32
// CHECK: [[PREV_VALUE:%.+]] = load atomic i8, i8* getelementptr inbounds (%struct.BitFields4_packed, %struct.BitFields4_packed* @{{.+}}, i32 0, i32 0, i64 2) monotonic
// CHECK: br label %[[CONT:.+]]
// CHECK: [[CONT]]
// CHECK: [[OLD_BF_VALUE:%.+]] = phi i8 [ [[PREV_VALUE]], %[[EXIT]] ], [ [[FAILED_OLD_VAL:%.+]], %[[CONT]] ]
// CHECK: [[TRUNC:%.+]] = trunc i32 [[NEW_VAL]] to i8
// CHECK: [[BF_VALUE:%.+]] = and i8 [[TRUNC]], 1
// CHECK: [[BF_CLEAR:%.+]] = and i8 %{{.+}}, -2
// CHECK: or i8 [[BF_CLEAR]], [[BF_VALUE]]
// CHECK: store i8 %{{.+}}, i8* [[LDTEMP:%.+]]
// CHECK: [[NEW_BF_VALUE:%.+]] = load i8, i8* [[LDTEMP]]
// CHECK: [[RES:%.+]] = cmpxchg i8* getelementptr inbounds (%struct.BitFields4_packed, %struct.BitFields4_packed* @{{.+}}, i32 0, i32 0, i64 2), i8 [[OLD_BF_VALUE]], i8 [[NEW_BF_VALUE]] monotonic monotonic
// CHECK: [[FAILED_OLD_VAL]] = extractvalue { i8, i1 } [[RES]], 0
// CHECK: [[FAIL_SUCCESS:%.+]] = extractvalue { i8, i1 } [[RES]], 1
// CHECK: br i1 [[FAIL_SUCCESS]], label %[[EXIT:.+]], label %[[CONT]]
// CHECK: [[EXIT]]
#pragma omp atomic write
  bfx4_packed.a = ldv;
// CHECK: load x86_fp80, x86_fp80* @{{.+}}
// CHECK: [[NEW_VAL:%.+]] = fptosi x86_fp80 %{{.+}} to i64
// CHECK: [[PREV_VALUE:%.+]] = load atomic i64, i64* bitcast (%struct.BitFields4* @{{.+}} to i64*) monotonic
// CHECK: br label %[[CONT:.+]]
// CHECK: [[CONT]]
// CHECK: [[OLD_BF_VALUE:%.+]] = phi i64 [ [[PREV_VALUE]], %[[EXIT]] ], [ [[FAILED_OLD_VAL:%.+]], %[[CONT]] ]
// CHECK: [[BF_AND:%.+]] = and i64 [[NEW_VAL]], 127
// CHECK: [[BF_VALUE:%.+]] = shl i64 [[BF_AND]], 17
// CHECK: [[BF_CLEAR:%.+]] = and i64 %{{.+}}, -16646145
// CHECK: or i64 [[BF_CLEAR]], [[BF_VALUE]]
// CHECK: store i64 %{{.+}}, i64* [[LDTEMP:%.+]]
// CHECK: [[NEW_BF_VALUE:%.+]] = load i64, i64* [[LDTEMP]]
// CHECK: [[RES:%.+]] = cmpxchg i64* bitcast (%struct.BitFields4* @{{.+}} to i64*), i64 [[OLD_BF_VALUE]], i64 [[NEW_BF_VALUE]] monotonic monotonic
// CHECK: [[FAILED_OLD_VAL]] = extractvalue { i64, i1 } [[RES]], 0
// CHECK: [[FAIL_SUCCESS:%.+]] = extractvalue { i64, i1 } [[RES]], 1
// CHECK: br i1 [[FAIL_SUCCESS]], label %[[EXIT:.+]], label %[[CONT]]
// CHECK: [[EXIT]]
#pragma omp atomic write
  bfx4.b = ldv;
// CHECK: load x86_fp80, x86_fp80* @{{.+}}
// CHECK: [[NEW_VAL:%.+]] = fptosi x86_fp80 %{{.+}} to i64
// CHECK: [[PREV_VALUE:%.+]] = load atomic i8, i8* getelementptr inbounds (%struct.BitFields4_packed, %struct.BitFields4_packed* @{{.+}}, i32 0, i32 0, i64 2) monotonic
// CHECK: br label %[[CONT:.+]]
// CHECK: [[CONT]]
// CHECK: [[OLD_BF_VALUE:%.+]] = phi i8 [ [[PREV_VALUE]], %[[EXIT]] ], [ [[FAILED_OLD_VAL:%.+]], %[[CONT]] ]
// CHECK: [[TRUNC:%.+]] = trunc i64 [[NEW_VAL]] to i8
// CHECK: [[BF_AND:%.+]] = and i8 [[TRUNC]], 127
// CHECK: [[BF_VALUE:%.+]] = shl i8 [[BF_AND]], 1
// CHECK: [[BF_CLEAR:%.+]] = and i8 %{{.+}}, 1
// CHECK: or i8 [[BF_CLEAR]], [[BF_VALUE]]
// CHECK: store i8 %{{.+}}, i8* [[LDTEMP:%.+]]
// CHECK: [[NEW_BF_VALUE:%.+]] = load i8, i8* [[LDTEMP]]
// CHECK: [[RES:%.+]] = cmpxchg i8* getelementptr inbounds (%struct.BitFields4_packed, %struct.BitFields4_packed* @{{.+}}, i32 0, i32 0, i64 2), i8 [[OLD_BF_VALUE]], i8 [[NEW_BF_VALUE]] monotonic monotonic
// CHECK: [[FAILED_OLD_VAL]] = extractvalue { i8, i1 } [[RES]], 0
// CHECK: [[FAIL_SUCCESS:%.+]] = extractvalue { i8, i1 } [[RES]], 1
// CHECK: br i1 [[FAIL_SUCCESS]], label %[[EXIT:.+]], label %[[CONT]]
// CHECK: [[EXIT]]
#pragma omp atomic write
  bfx4_packed.b = ldv;
// CHECK: load i64, i64*
// CHECK: [[VEC_ITEM_VAL:%.+]] = uitofp i64 %{{.+}} to float
// CHECK: [[I64VAL:%.+]] = load atomic i64, i64* bitcast (<2 x float>* [[DEST:@.+]] to i64*) monotonic
// CHECK: br label %[[CONT:.+]]
// CHECK: [[CONT]]
// CHECK: [[OLD_I64:%.+]] = phi i64 [ [[I64VAL]], %{{.+}} ], [ [[FAILED_I64_OLD_VAL:%.+]], %[[CONT]] ]
// CHECK: [[BITCAST:%.+]] = bitcast <2 x float>* [[LDTEMP:%.+]] to i64*
// CHECK: store i64 [[OLD_I64]], i64* [[BITCAST]],
// CHECK: [[VEC_VAL:%.+]] = load <2 x float>, <2 x float>* [[LDTEMP]]
// CHECK: [[NEW_VEC_VAL:%.+]] = insertelement <2 x float> [[VEC_VAL]], float [[VEC_ITEM_VAL]], i64 0
// CHECK: store <2 x float> [[NEW_VEC_VAL]], <2 x float>* [[LDTEMP]]
// CHECK: [[NEW_I64:%.+]] = load i64, i64* [[BITCAST]]
// CHECK: [[RES:%.+]] = cmpxchg i64* bitcast (<2 x float>* [[DEST]] to i64*), i64 [[OLD_I64]], i64 [[NEW_I64]] monotonic monotonic
// CHECK: [[FAILED_I64_OLD_VAL:%.+]] = extractvalue { i64, i1 } [[RES]], 0
// CHECK: [[FAIL_SUCCESS:%.+]] = extractvalue { i64, i1 } [[RES]], 1
// CHECK: br i1 [[FAIL_SUCCESS]], label %[[EXIT:.+]], label %[[CONT]]
// CHECK: [[EXIT]]
#pragma omp atomic write
  float2x.x = ulv;
// CHECK: call i32 @llvm.read_register.i32(
// CHECK: sitofp i32 %{{.+}} to double
// CHECK: bitcast double %{{.+}} to i64
// CHECK: store atomic i64 %{{.+}}, i64* bitcast (double* @{{.+}} to i64*) seq_cst
// CHECK: call{{.*}} @__kmpc_flush(
#pragma omp atomic write seq_cst
  dv = rix;
  return 0;
}

#endif

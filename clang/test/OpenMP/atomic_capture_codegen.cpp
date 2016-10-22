// RUN: %clang_cc1 -verify -triple x86_64-apple-darwin10 -fopenmp -x c -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -fopenmp -x c -triple x86_64-apple-darwin10 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -x c -triple x86_64-apple-darwin10 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s
// expected-no-diagnostics
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
// CHECK: [[PREV:%.+]] = atomicrmw add i8* @{{.+}}, i8 1 monotonic
// CHECK: store i8 [[PREV]], i8* @{{.+}},
#pragma omp atomic capture
  bv = bx++;
// CHECK: atomicrmw add i8* @{{.+}}, i8 1 monotonic
// CHECK: add nsw i32 %{{.+}}, 1
// CHECK: store i8 %{{.+}}, i8* @{{.+}},
#pragma omp atomic capture
  cv = ++cx;
// CHECK: [[PREV:%.+]] = atomicrmw sub i8* @{{.+}}, i8 1 monotonic
// CHECK: store i8 [[PREV]], i8* @{{.+}},
#pragma omp atomic capture
  ucv = ucx--;
// CHECK: atomicrmw sub i16* @{{.+}}, i16 1 monotonic
// CHECK: sub nsw i32 %{{.+}}, 1
// CHECK: store i16 %{{.+}}, i16* @{{.+}},
#pragma omp atomic capture
  sv = --sx;
// CHECK: [[USV:%.+]] = load i16, i16* @{{.+}},
// CHECK: [[EXPR:%.+]] = zext i16 [[USV]] to i32
// CHECK: [[X:%.+]] = load atomic i16, i16* [[X_ADDR:@.+]] monotonic
// CHECK: br label %[[CONT:.+]]
// CHECK: [[CONT]]
// CHECK: [[EXPECTED:%.+]] = phi i16 [ [[X]], %{{.+}} ], [ [[OLD_X:%.+]], %[[CONT]] ]
// CHECK: [[CONV:%.+]] = zext i16 [[EXPECTED]] to i32
// CHECK: [[ADD:%.+]] = add nsw i32 [[CONV]], [[EXPR]]
// CHECK: [[DESIRED_CALC:%.+]] = trunc i32 [[ADD]] to i16
// CHECK: store i16 [[DESIRED_CALC]], i16* [[TEMP:%.+]],
// CHECK: [[DESIRED:%.+]] = load i16, i16* [[TEMP]],
// CHECK: [[RES:%.+]] = cmpxchg i16* [[X_ADDR]], i16 [[EXPECTED]], i16 [[DESIRED]] monotonic monotonic
// CHECK: [[OLD_X]] = extractvalue { i16, i1 } [[RES]], 0
// CHECK: [[SUCCESS_FAIL:%.+]] = extractvalue { i16, i1 } [[RES]], 1
// CHECK: br i1 [[SUCCESS_FAIL]], label %[[EXIT:.+]], label %[[CONT]]
// CHECK: [[EXIT]]
// CHECK: store i16 [[DESIRED_CALC]], i16* @{{.+}},
#pragma omp atomic capture
  sv = usx += usv;
// CHECK: [[EXPR:%.+]] = load i32, i32* @{{.+}},
// CHECK: [[X:%.+]] = load atomic i32, i32* [[X_ADDR:@.+]] monotonic
// CHECK: br label %[[CONT:.+]]
// CHECK: [[CONT]]
// CHECK: [[EXPECTED:%.+]] = phi i32 [ [[X]], %{{.+}} ], [ [[OLD_X:%.+]], %[[CONT]] ]
// CHECK: [[DESIRED_CALC:%.+]] = mul nsw i32 [[EXPECTED]], [[EXPR]]
// CHECK: store i32 [[DESIRED_CALC]], i32* [[TEMP:%.+]],
// CHECK: [[DESIRED:%.+]] = load i32, i32* [[TEMP]],
// CHECK: [[RES:%.+]] = cmpxchg i32* [[X_ADDR]], i32 [[EXPECTED]], i32 [[DESIRED]] monotonic monotonic
// CHECK: [[OLD_X]] = extractvalue { i32, i1 } [[RES]], 0
// CHECK: [[SUCCESS_FAIL:%.+]] = extractvalue { i32, i1 } [[RES]], 1
// CHECK: br i1 [[SUCCESS_FAIL]], label %[[EXIT:.+]], label %[[CONT]]
// CHECK: [[EXIT]]
// CHECK: store i32 [[DESIRED_CALC]], i32* @{{.+}},
#pragma omp atomic capture
  uiv = ix *= iv;
// CHECK: [[EXPR:%.+]] = load i32, i32* @{{.+}},
// CHECK: [[PREV:%.+]] = atomicrmw sub i32* @{{.+}}, i32 [[EXPR]] monotonic
// CHECK: store i32 [[PREV]], i32* @{{.+}},
#pragma omp atomic capture
  {iv = uix; uix -= uiv;}
// CHECK: [[EXPR:%.+]] = load i32, i32* @{{.+}},
// CHECK: [[X:%.+]] = load atomic i32, i32* [[X_ADDR:@.+]] monotonic
// CHECK: br label %[[CONT:.+]]
// CHECK: [[CONT]]
// CHECK: [[EXPECTED:%.+]] = phi i32 [ [[X]], %{{.+}} ], [ [[OLD_X:%.+]], %[[CONT]] ]
// CHECK: [[DESIRED_CALC:%.+]] = shl i32 [[EXPECTED]], [[EXPR]]
// CHECK: store i32 [[DESIRED_CALC]], i32* [[TEMP:%.+]],
// CHECK: [[DESIRED:%.+]] = load i32, i32* [[TEMP]],
// CHECK: [[RES:%.+]] = cmpxchg i32* [[X_ADDR]], i32 [[EXPECTED]], i32 [[DESIRED]] monotonic monotonic
// CHECK: [[OLD_X]] = extractvalue { i32, i1 } [[RES]], 0
// CHECK: [[SUCCESS_FAIL:%.+]] = extractvalue { i32, i1 } [[RES]], 1
// CHECK: br i1 [[SUCCESS_FAIL]], label %[[EXIT:.+]], label %[[CONT]]
// CHECK: [[EXIT]]
// CHECK: store i32 [[DESIRED_CALC]], i32* @{{.+}},
#pragma omp atomic capture
  {ix <<= iv; uiv = ix;}
// CHECK: [[EXPR:%.+]] = load i32, i32* @{{.+}},
// CHECK: [[X:%.+]] = load atomic i32, i32* [[X_ADDR:@.+]] monotonic
// CHECK: br label %[[CONT:.+]]
// CHECK: [[CONT]]
// CHECK: [[EXPECTED:%.+]] = phi i32 [ [[X]], %{{.+}} ], [ [[OLD_X:%.+]], %[[CONT]] ]
// CHECK: [[DESIRED_CALC:%.+]] = lshr i32 [[EXPECTED]], [[EXPR]]
// CHECK: store i32 [[DESIRED_CALC]], i32* [[TEMP:%.+]],
// CHECK: [[DESIRED:%.+]] = load i32, i32* [[TEMP]],
// CHECK: [[RES:%.+]] = cmpxchg i32* [[X_ADDR]], i32 [[EXPECTED]], i32 [[DESIRED]] monotonic monotonic
// CHECK: [[OLD_X]] = extractvalue { i32, i1 } [[RES]], 0
// CHECK: [[SUCCESS_FAIL:%.+]] = extractvalue { i32, i1 } [[RES]], 1
// CHECK: br i1 [[SUCCESS_FAIL]], label %[[EXIT:.+]], label %[[CONT]]
// CHECK: [[EXIT]]
// CHECK: store i32 [[DESIRED_CALC]], i32* @{{.+}},
#pragma omp atomic capture
  iv = uix >>= uiv;
// CHECK: [[EXPR:%.+]] = load i64, i64* @{{.+}},
// CHECK: [[X:%.+]] = load atomic i64, i64* [[X_ADDR:@.+]] monotonic
// CHECK: br label %[[CONT:.+]]
// CHECK: [[CONT]]
// CHECK: [[EXPECTED:%.+]] = phi i64 [ [[X]], %{{.+}} ], [ [[OLD_X:%.+]], %[[CONT]] ]
// CHECK: [[DESIRED:%.+]] = sdiv i64 [[EXPECTED]], [[EXPR]]
// CHECK: store i64 [[DESIRED]], i64* [[TEMP:%.+]],
// CHECK: [[DESIRED:%.+]] = load i64, i64* [[TEMP]],
// CHECK: [[RES:%.+]] = cmpxchg i64* [[X_ADDR]], i64 [[EXPECTED]], i64 [[DESIRED]] monotonic monotonic
// CHECK: [[OLD_X]] = extractvalue { i64, i1 } [[RES]], 0
// CHECK: [[SUCCESS_FAIL:%.+]] = extractvalue { i64, i1 } [[RES]], 1
// CHECK: br i1 [[SUCCESS_FAIL]], label %[[EXIT:.+]], label %[[CONT]]
// CHECK: [[EXIT]]
// CHECK: store i64 [[EXPECTED]], i64* @{{.+}},
#pragma omp atomic capture
  {ulv = lx; lx /= lv;}
// CHECK: [[EXPR:%.+]] = load i64, i64* @{{.+}},
// CHECK: [[OLD:%.+]] = atomicrmw and i64* @{{.+}}, i64 [[EXPR]] monotonic
// CHECK: [[DESIRED:%.+]] = and i64 [[OLD]], [[EXPR]]
// CHECK:  store i64 [[DESIRED]], i64* @{{.+}},
#pragma omp atomic capture
  {ulx &= ulv; lv = ulx;}
// CHECK: [[EXPR:%.+]] = load i64, i64* @{{.+}},
// CHECK: [[OLD:%.+]] = atomicrmw xor i64* @{{.+}}, i64 [[EXPR]] monotonic
// CHECK: [[DESIRED:%.+]] = xor i64 [[OLD]], [[EXPR]]
// CHECK:  store i64 [[DESIRED]], i64* @{{.+}},
#pragma omp atomic capture
  ullv = llx ^= llv;
// CHECK: [[EXPR:%.+]] = load i64, i64* @{{.+}},
// CHECK: [[OLD:%.+]] = atomicrmw or i64* @{{.+}}, i64 [[EXPR]] monotonic
// CHECK: [[DESIRED:%.+]] = or i64 [[OLD]], [[EXPR]]
// CHECK:  store i64 [[DESIRED]], i64* @{{.+}},
#pragma omp atomic capture
  llv = ullx |= ullv;
// CHECK: [[EXPR:%.+]] = load float, float* @{{.+}},
// CHECK: [[X:%.+]] = load atomic i32, i32*  bitcast (float* [[X_ADDR:@.+]] to i32*) monotonic
// CHECK: br label %[[CONT:.+]]
// CHECK: [[CONT]]
// CHECK: [[EXPECTED:%.+]] = phi i32 [ [[X]], %{{.+}} ], [ [[OLD_X:%.+]], %[[CONT]] ]
// CHECK: [[TEMP_I:%.+]] = bitcast float* [[TEMP:%.+]] to i32*
// CHECK: [[OLD:%.+]] = bitcast i32 [[EXPECTED]] to float
// CHECK: [[ADD:%.+]] = fadd float [[OLD]], [[EXPR]]
// CHECK: store float [[ADD]], float* [[TEMP]],
// CHECK: [[DESIRED:%.+]] = load i32, i32* [[TEMP_I]],
// CHECK: [[RES:%.+]] = cmpxchg i32* bitcast (float* [[X_ADDR]] to i32*), i32 [[EXPECTED]], i32 [[DESIRED]] monotonic monotonic
// CHECK: [[OLD_X:%.+]] = extractvalue { i32, i1 } [[RES]], 0
// CHECK: [[SUCCESS_FAIL:%.+]] = extractvalue { i32, i1 } [[RES]], 1
// CHECK: br i1 [[SUCCESS_FAIL]], label %[[EXIT:.+]], label %[[CONT]]
// CHECK: [[EXIT]]
// CHECK: [[CAST:%.+]] = fpext float [[ADD]] to double
// CHECK: store double [[CAST]], double* @{{.+}},
#pragma omp atomic capture
  dv = fx = fx + fv;
// CHECK: [[EXPR:%.+]] = load double, double* @{{.+}},
// CHECK: [[X:%.+]] = load atomic i64, i64*  bitcast (double* [[X_ADDR:@.+]] to i64*) monotonic
// CHECK: br label %[[CONT:.+]]
// CHECK: [[CONT]]
// CHECK: [[EXPECTED:%.+]] = phi i64 [ [[X]], %{{.+}} ], [ [[OLD_X:%.+]], %[[CONT]] ]
// CHECK: [[TEMP_I:%.+]] = bitcast double* [[TEMP:%.+]] to i64*
// CHECK: [[OLD:%.+]] = bitcast i64 [[EXPECTED]] to double
// CHECK: [[SUB:%.+]] = fsub double [[EXPR]], [[OLD]]
// CHECK: store double [[SUB]], double* [[TEMP]],
// CHECK: [[DESIRED:%.+]] = load i64, i64* [[TEMP_I]],
// CHECK: [[RES:%.+]] = cmpxchg i64* bitcast (double* [[X_ADDR]] to i64*), i64 [[EXPECTED]], i64 [[DESIRED]] monotonic monotonic
// CHECK: [[OLD_X:%.+]] = extractvalue { i64, i1 } [[RES]], 0
// CHECK: [[SUCCESS_FAIL:%.+]] = extractvalue { i64, i1 } [[RES]], 1
// CHECK: br i1 [[SUCCESS_FAIL]], label %[[EXIT:.+]], label %[[CONT]]
// CHECK: [[EXIT]]
// CHECK: [[CAST:%.+]] = fptrunc double [[OLD]] to float
// CHECK: store float [[CAST]], float* @{{.+}},
#pragma omp atomic capture
  {fv = dx; dx = dv - dx;}
// CHECK: [[EXPR:%.+]] = load x86_fp80, x86_fp80* @{{.+}},
// CHECK: [[X:%.+]] = load atomic i128, i128*  bitcast (x86_fp80* [[X_ADDR:@.+]] to i128*) monotonic
// CHECK: br label %[[CONT:.+]]
// CHECK: [[CONT]]
// CHECK: [[EXPECTED:%.+]] = phi i128 [ [[X]], %{{.+}} ], [ [[OLD_X:%.+]], %[[CONT]] ]
// CHECK: [[BITCAST:%.+]] = bitcast x86_fp80* [[TEMP:%.+]] to i128*
// CHECK: store i128 [[EXPECTED]], i128* [[BITCAST]]
// CHECK: [[BITCAST1:%.+]] = bitcast x86_fp80* [[TEMP1:%.+]] to i128*
// CHECK: store i128 [[EXPECTED]], i128* [[BITCAST1]]
// CHECK: [[OLD:%.+]] = load x86_fp80, x86_fp80* [[TEMP1]]
// CHECK: [[MUL:%.+]] = fmul x86_fp80 [[OLD]], [[EXPR]]
// CHECK: store x86_fp80 [[MUL]], x86_fp80* [[TEMP]]
// CHECK: [[DESIRED:%.+]] = load i128, i128* [[BITCAST]]
// CHECK: [[RES:%.+]] = cmpxchg i128* bitcast (x86_fp80* [[X_ADDR]] to i128*), i128 [[EXPECTED]], i128 [[DESIRED]] monotonic monotonic
// CHECK: [[OLD_X:%.+]] = extractvalue { i128, i1 } [[RES]], 0
// CHECK: [[SUCCESS_FAIL:%.+]] = extractvalue { i128, i1 } [[RES]], 1
// CHECK: br i1 [[SUCCESS_FAIL]], label %[[EXIT:.+]], label %[[CONT]]
// CHECK: [[EXIT]]
// CHECK: [[CAST:%.+]] = fptrunc x86_fp80 [[MUL]] to double
// CHECK: store double [[CAST]], double* @{{.+}},
#pragma omp atomic capture
  {ldx = ldx * ldv; dv = ldx;}
// CHECK: [[EXPR_RE:%.+]] = load i32, i32* getelementptr inbounds ({ i32, i32 }, { i32, i32 }* @{{.+}}, i32 0, i32 0)
// CHECK: [[EXPR_IM:%.+]] = load i32, i32* getelementptr inbounds ({ i32, i32 }, { i32, i32 }* @{{.+}}, i32 0, i32 1)
// CHECK: [[BITCAST:%.+]] = bitcast { i32, i32 }* [[EXPECTED_ADDR:%.+]] to i8*
// CHECK: call void @__atomic_load(i64 8, i8* bitcast ({ i32, i32 }* [[X_ADDR:@.+]] to i8*), i8* [[BITCAST]], i32 0)
// CHECK: br label %[[CONT:.+]]
// CHECK: [[CONT]]
// CHECK: [[LD_RE_ADDR:%.+]] = getelementptr inbounds { i32, i32 }, { i32, i32 }* [[EXPECTED_ADDR]], i32 0, i32 0
// CHECK: [[LD_RE:%.+]] = load i32, i32* [[LD_RE_ADDR]]
// CHECK: [[LD_IM_ADDR:%.+]] = getelementptr inbounds { i32, i32 }, { i32, i32 }* [[EXPECTED_ADDR]], i32 0, i32 1
// CHECK: [[LD_IM:%.+]] = load i32, i32* [[LD_IM_ADDR]]
// <Skip checks for complex calculations>
// CHECK: [[X_RE_ADDR:%.+]] = getelementptr inbounds { i32, i32 }, { i32, i32 }* [[DESIRED_ADDR:%.+]], i32 0, i32 0
// CHECK: [[X_IM_ADDR:%.+]] = getelementptr inbounds { i32, i32 }, { i32, i32 }* [[DESIRED_ADDR]], i32 0, i32 1
// CHECK: store i32 [[NEW_RE:%.+]], i32* [[X_RE_ADDR]]
// CHECK: store i32 [[NEW_IM:%.+]], i32* [[X_IM_ADDR]]
// CHECK: [[EXPECTED:%.+]] = bitcast { i32, i32 }* [[EXPECTED_ADDR]] to i8*
// CHECK: [[DESIRED:%.+]] = bitcast { i32, i32 }* [[DESIRED_ADDR]] to i8*
// CHECK: [[SUCCESS_FAIL:%.+]] = call zeroext i1 @__atomic_compare_exchange(i64 8, i8* bitcast ({ i32, i32 }* [[X_ADDR]] to i8*), i8* [[EXPECTED]], i8* [[DESIRED]], i32 0, i32 0)
// CHECK: br i1 [[SUCCESS_FAIL]], label %[[EXIT:.+]], label %[[CONT]]
// CHECK: [[EXIT]]
// CHECK: [[RE_CAST:%.+]] = sitofp i32 [[NEW_RE]] to float
// CHECK: [[IM_CAST:%.+]] = sitofp i32 [[NEW_IM]] to float
// CHECK: store float [[RE_CAST]], float* getelementptr inbounds ({ float, float }, { float, float }* @{{.+}}, i32 0, i32 0),
// CHECK: store float [[IM_CAST]], float* getelementptr inbounds ({ float, float }, { float, float }* @{{.+}}, i32 0, i32 1),
#pragma omp atomic capture
  cfv = cix = civ / cix;
// CHECK: [[EXPR_RE:%.+]] = load float, float* getelementptr inbounds ({ float, float }, { float, float }* @{{.+}}, i32 0, i32 0)
// CHECK: [[EXPR_IM:%.+]] = load float, float* getelementptr inbounds ({ float, float }, { float, float }* @{{.+}}, i32 0, i32 1)
// CHECK: [[BITCAST:%.+]] = bitcast { float, float }* [[EXPECTED_ADDR:%.+]] to i8*
// CHECK: call void @__atomic_load(i64 8, i8* bitcast ({ float, float }* [[X_ADDR:@.+]] to i8*), i8* [[BITCAST]], i32 0)
// CHECK: br label %[[CONT:.+]]
// CHECK: [[CONT]]
// CHECK: [[X_RE_ADDR:%.+]] = getelementptr inbounds { float, float }, { float, float }* [[EXPECTED_ADDR]], i32 0, i32 0
// CHECK: [[X_RE_OLD:%.+]] = load float, float* [[X_RE_ADDR]]
// CHECK: [[X_IM_ADDR:%.+]] = getelementptr inbounds { float, float }, { float, float }* [[EXPECTED_ADDR]], i32 0, i32 1
// CHECK: [[X_IM_OLD:%.+]] = load float, float* [[X_IM_ADDR]]
// <Skip checks for complex calculations>
// CHECK: [[X_RE_ADDR:%.+]] = getelementptr inbounds { float, float }, { float, float }* [[DESIRED_ADDR:%.+]], i32 0, i32 0
// CHECK: [[X_IM_ADDR:%.+]] = getelementptr inbounds { float, float }, { float, float }* [[DESIRED_ADDR]], i32 0, i32 1
// CHECK: store float [[NEW_RE:%.+]], float* [[X_RE_ADDR]]
// CHECK: store float [[NEW_IM:%.+]], float* [[X_IM_ADDR]]
// CHECK: [[EXPECTED:%.+]] = bitcast { float, float }* [[EXPECTED_ADDR]] to i8*
// CHECK: [[DESIRED:%.+]] = bitcast { float, float }* [[DESIRED_ADDR]] to i8*
// CHECK: [[SUCCESS_FAIL:%.+]] = call zeroext i1 @__atomic_compare_exchange(i64 8, i8* bitcast ({ float, float }* [[X_ADDR]] to i8*), i8* [[EXPECTED]], i8* [[DESIRED]], i32 0, i32 0)
// CHECK: br i1 [[SUCCESS_FAIL]], label %[[EXIT:.+]], label %[[CONT]]
// CHECK: [[EXIT]]
// CHECK: [[RE_CAST:%.+]] = fptosi float [[X_RE_OLD]] to i32
// CHECK: [[IM_CAST:%.+]] = fptosi float [[X_IM_OLD]] to i32
// CHECK: store i32 [[RE_CAST]], i32* getelementptr inbounds ({ i32, i32 }, { i32, i32 }* @{{.+}}, i32 0, i32 0),
// CHECK: store i32 [[IM_CAST]], i32* getelementptr inbounds ({ i32, i32 }, { i32, i32 }* @{{.+}}, i32 0, i32 1),
#pragma omp atomic capture
  {civ = cfx; cfx = cfv + cfx;}
// CHECK: [[EXPR_RE:%.+]] = load double, double* getelementptr inbounds ({ double, double }, { double, double }* @{{.+}}, i32 0, i32 0)
// CHECK: [[EXPR_IM:%.+]] = load double, double* getelementptr inbounds ({ double, double }, { double, double }* @{{.+}}, i32 0, i32 1)
// CHECK: [[BITCAST:%.+]] = bitcast { double, double }* [[EXPECTED_ADDR:%.+]] to i8*
// CHECK: call void @__atomic_load(i64 16, i8* bitcast ({ double, double }* [[X_ADDR:@.+]] to i8*), i8* [[BITCAST]], i32 5)
// CHECK: br label %[[CONT:.+]]
// CHECK: [[CONT]]
// CHECK: [[X_RE_ADDR:%.+]] = getelementptr inbounds { double, double }, { double, double }* [[EXPECTED_ADDR]], i32 0, i32 0
// CHECK: [[X_RE:%.+]] = load double, double* [[X_RE_ADDR]]
// CHECK: [[X_IM_ADDR:%.+]] = getelementptr inbounds { double, double }, { double, double }* [[EXPECTED_ADDR]], i32 0, i32 1
// CHECK: [[X_IM:%.+]] = load double, double* [[X_IM_ADDR]]
// <Skip checks for complex calculations>
// CHECK: [[X_RE_ADDR:%.+]] = getelementptr inbounds { double, double }, { double, double }* [[DESIRED_ADDR:%.+]], i32 0, i32 0
// CHECK: [[X_IM_ADDR:%.+]] = getelementptr inbounds { double, double }, { double, double }* [[DESIRED_ADDR]], i32 0, i32 1
// CHECK: store double [[NEW_RE:%.+]], double* [[X_RE_ADDR]]
// CHECK: store double [[NEW_IM:%.+]], double* [[X_IM_ADDR]]
// CHECK: [[EXPECTED:%.+]] = bitcast { double, double }* [[EXPECTED_ADDR]] to i8*
// CHECK: [[DESIRED:%.+]] = bitcast { double, double }* [[DESIRED_ADDR]] to i8*
// CHECK: [[SUCCESS_FAIL:%.+]] = call zeroext i1 @__atomic_compare_exchange(i64 16, i8* bitcast ({ double, double }* [[X_ADDR]] to i8*), i8* [[EXPECTED]], i8* [[DESIRED]], i32 5, i32 5)
// CHECK: br i1 [[SUCCESS_FAIL]], label %[[EXIT:.+]], label %[[CONT]]
// CHECK: [[EXIT]]
// CHECK: [[RE_CAST:%.+]] = fptrunc double [[NEW_RE]] to float
// CHECK: [[IM_CAST:%.+]] = fptrunc double [[NEW_IM]] to float
// CHECK: store float [[RE_CAST]], float* getelementptr inbounds ({ float, float }, { float, float }* @{{.+}}, i32 0, i32 0),
// CHECK: store float [[IM_CAST]], float* getelementptr inbounds ({ float, float }, { float, float }* @{{.+}}, i32 0, i32 1),
// CHECK: call{{.*}} @__kmpc_flush(
#pragma omp atomic capture seq_cst
  {cdx = cdx - cdv; cfv = cdx;}
// CHECK: [[BV:%.+]] = load i8, i8* @{{.+}}
// CHECK: [[BOOL:%.+]] = trunc i8 [[BV]] to i1
// CHECK: [[EXPR:%.+]] = zext i1 [[BOOL]] to i64
// CHECK: [[OLD:%.+]] = atomicrmw and i64* @{{.+}}, i64 [[EXPR]] monotonic
// CHECK: [[DESIRED:%.+]] = and i64 [[OLD]], [[EXPR]]
// CHECK: store i64 [[DESIRED]], i64* @{{.+}},
#pragma omp atomic capture
  ulv = ulx = ulx & bv;
// CHECK: [[CV:%.+]]  = load i8, i8* @{{.+}}, align 1
// CHECK: [[EXPR:%.+]] = sext i8 [[CV]] to i32
// CHECK: [[X:%.+]] = load atomic i8, i8* [[BX_ADDR:@.+]] monotonic
// CHECK: br label %[[CONT:.+]]
// CHECK: [[CONT]]
// CHECK: [[EXPECTED:%.+]] = phi i8 [ [[X]], %{{.+}} ], [ [[OLD_X:%.+]], %[[CONT]] ]
// CHECK: [[OLD_BOOL:%.+]] = trunc i8 [[EXPECTED]] to i1
// CHECK: [[X_RVAL:%.+]] = zext i1 [[OLD_BOOL]] to i32
// CHECK: [[AND:%.+]] = and i32 [[EXPR]], [[X_RVAL]]
// CHECK: [[CAST:%.+]] = icmp ne i32 [[AND]], 0
// CHECK: [[NEW:%.+]] = zext i1 [[CAST]] to i8
// CHECK: store i8 [[NEW]], i8* [[TEMP:%.+]],
// CHECK: [[DESIRED:%.+]] = load i8, i8* [[TEMP]],
// CHECK: [[RES:%.+]] = cmpxchg i8* [[BX_ADDR]], i8 [[EXPECTED]], i8 [[DESIRED]] monotonic monotonic
// CHECK: [[OLD:%.+]] = extractvalue { i8, i1 } [[RES]], 0
// CHECK: [[SUCCESS_FAIL:%.+]] = extractvalue { i8, i1 } [[RES]], 1
// CHECK: br i1 [[SUCCESS_FAIL]], label %[[EXIT:.+]], label %[[CONT]]
// CHECK: [[EXIT]]
// CHECK: [[OLD_I8:%.+]] = zext i1 [[OLD_BOOL]] to i8
// CHECK: store i8 [[OLD_I8]], i8* @{{.+}},
#pragma omp atomic capture
  {bv = bx; bx = cv & bx;}
// CHECK: [[UCV:%.+]]  = load i8, i8* @{{.+}},
// CHECK: [[EXPR:%.+]] = zext i8 [[UCV]] to i32
// CHECK: [[X:%.+]] = load atomic i8, i8* [[CX_ADDR:@.+]] seq_cst
// CHECK: br label %[[CONT:.+]]
// CHECK: [[CONT]]
// CHECK: [[EXPECTED:%.+]] = phi i8 [ [[X]], %{{.+}} ], [ [[OLD_X:%.+]], %[[CONT]] ]
// CHECK: [[X_RVAL:%.+]] = sext i8 [[EXPECTED]] to i32
// CHECK: [[ASHR:%.+]] = ashr i32 [[X_RVAL]], [[EXPR]]
// CHECK: [[NEW:%.+]] = trunc i32 [[ASHR]] to i8
// CHECK: store i8 [[NEW]], i8* [[TEMP:%.+]],
// CHECK: [[DESIRED:%.+]] = load i8, i8* [[TEMP]],
// CHECK: [[RES:%.+]] = cmpxchg i8* [[CX_ADDR]], i8 [[EXPECTED]], i8 [[DESIRED]] seq_cst seq_cst
// CHECK: [[OLD_X:%.+]] = extractvalue { i8, i1 } [[RES]], 0
// CHECK: [[SUCCESS_FAIL:%.+]] = extractvalue { i8, i1 } [[RES]], 1
// CHECK: br i1 [[SUCCESS_FAIL]], label %[[EXIT:.+]], label %[[CONT]]
// CHECK: [[EXIT]]
// CHECK: store i8 [[NEW]], i8* @{{.+}},
// CHECK: call{{.*}} @__kmpc_flush(
#pragma omp atomic capture, seq_cst
  {cx = cx >> ucv; cv = cx;}
// CHECK: [[SV:%.+]]  = load i16, i16* @{{.+}},
// CHECK: [[EXPR:%.+]] = sext i16 [[SV]] to i32
// CHECK: [[X:%.+]] = load atomic i64, i64* [[ULX_ADDR:@.+]] monotonic
// CHECK: br label %[[CONT:.+]]
// CHECK: [[CONT]]
// CHECK: [[EXPECTED:%.+]] = phi i64 [ [[X]], %{{.+}} ], [ [[OLD_X:%.+]], %[[CONT]] ]
// CHECK: [[X_RVAL:%.+]] = trunc i64 [[EXPECTED]] to i32
// CHECK: [[SHL:%.+]] = shl i32 [[EXPR]], [[X_RVAL]]
// CHECK: [[NEW:%.+]] = sext i32 [[SHL]] to i64
// CHECK: store i64 [[NEW]], i64* [[TEMP:%.+]],
// CHECK: [[DESIRED:%.+]] = load i64, i64* [[TEMP]],
// CHECK: [[RES:%.+]] = cmpxchg i64* [[ULX_ADDR]], i64 [[EXPECTED]], i64 [[DESIRED]] monotonic monotonic
// CHECK: [[OLD_X:%.+]] = extractvalue { i64, i1 } [[RES]], 0
// CHECK: [[SUCCESS_FAIL:%.+]] = extractvalue { i64, i1 } [[RES]], 1
// CHECK: br i1 [[SUCCESS_FAIL]], label %[[EXIT:.+]], label %[[CONT]]
// CHECK: [[EXIT]]
// CHECK: store i64 [[NEW]], i64* @{{.+}},
#pragma omp atomic capture
  ulv = ulx = sv << ulx;
// CHECK: [[USV:%.+]]  = load i16, i16* @{{.+}},
// CHECK: [[EXPR:%.+]] = zext i16 [[USV]] to i64
// CHECK: [[X:%.+]] = load atomic i64, i64* [[LX_ADDR:@.+]] monotonic
// CHECK: br label %[[CONT:.+]]
// CHECK: [[CONT]]
// CHECK: [[EXPECTED:%.+]] = phi i64 [ [[X]], %{{.+}} ], [ [[OLD_X:%.+]], %[[CONT]] ]
// CHECK: [[DESIRED:%.+]] = srem i64 [[EXPECTED]], [[EXPR]]
// CHECK: store i64 [[DESIRED]], i64* [[TEMP:%.+]],
// CHECK: [[DESIRED:%.+]] = load i64, i64* [[TEMP]],
// CHECK: [[RES:%.+]] = cmpxchg i64* [[LX_ADDR]], i64 [[EXPECTED]], i64 [[DESIRED]] monotonic monotonic
// CHECK: [[OLD_X:%.+]] = extractvalue { i64, i1 } [[RES]], 0
// CHECK: [[SUCCESS_FAIL:%.+]] = extractvalue { i64, i1 } [[RES]], 1
// CHECK: br i1 [[SUCCESS_FAIL]], label %[[EXIT:.+]], label %[[CONT]]
// CHECK: [[EXIT]]
// CHECK: store i64 [[EXPECTED]], i64* @{{.+}},
#pragma omp atomic capture
  {lv = lx; lx = lx % usv;}
// CHECK: [[EXPR:%.+]] = load i32, i32* @{{.+}}
// CHECK: [[OLD:%.+]] = atomicrmw or i32* @{{.+}}, i32 [[EXPR]] seq_cst
// CHECK: [[DESIRED:%.+]] = or i32 [[EXPR]], [[OLD]]
// CHECK: store i32 [[DESIRED]], i32* @{{.+}},
// CHECK: call{{.*}} @__kmpc_flush(
#pragma omp atomic seq_cst, capture
  {uix = iv | uix; uiv = uix;}
// CHECK: [[EXPR:%.+]] = load i32, i32* @{{.+}}
// CHECK: [[OLD:%.+]] = atomicrmw and i32* @{{.+}}, i32 [[EXPR]] monotonic
// CHECK: [[DESIRED:%.+]] = and i32 [[OLD]], [[EXPR]]
// CHECK: store i32 [[DESIRED]], i32* @{{.+}},
#pragma omp atomic capture
  iv = ix = ix & uiv;
// CHECK: [[EXPR:%.+]] = load i64, i64* @{{.+}},
// CHECK: [[BITCAST:%.+]] = bitcast { i32, i32 }* [[EXPECTED_ADDR:%.+]] to i8*
// CHECK: call void @__atomic_load(i64 8, i8* bitcast ({ i32, i32 }* [[X_ADDR:@.+]] to i8*), i8* [[BITCAST]], i32 0)
// CHECK: br label %[[CONT:.+]]
// CHECK: [[CONT]]
// CHECK: [[X_RE_ADDR:%.+]] = getelementptr inbounds { i32, i32 }, { i32, i32 }* [[EXPECTED_ADDR]], i32 0, i32 0
// CHECK: [[OLD_RE:%.+]] = load i32, i32* [[X_RE_ADDR]]
// CHECK: [[X_IM_ADDR:%.+]] = getelementptr inbounds { i32, i32 }, { i32, i32 }* [[EXPECTED_ADDR]], i32 0, i32 1
// CHECK: [[OLD_IM:%.+]] = load i32, i32* [[X_IM_ADDR]]
// <Skip checks for complex calculations>
// CHECK: [[X_RE_ADDR:%.+]] = getelementptr inbounds { i32, i32 }, { i32, i32 }* [[DESIRED_ADDR:%.+]], i32 0, i32 0
// CHECK: [[X_IM_ADDR:%.+]] = getelementptr inbounds { i32, i32 }, { i32, i32 }* [[DESIRED_ADDR]], i32 0, i32 1
// CHECK: store i32 %{{.+}}, i32* [[X_RE_ADDR]]
// CHECK: store i32 %{{.+}}, i32* [[X_IM_ADDR]]
// CHECK: [[EXPECTED:%.+]] = bitcast { i32, i32 }* [[EXPECTED_ADDR]] to i8*
// CHECK: [[DESIRED:%.+]] = bitcast { i32, i32 }* [[DESIRED_ADDR]] to i8*
// CHECK: [[SUCCESS_FAIL:%.+]] = call zeroext i1 @__atomic_compare_exchange(i64 8, i8* bitcast ({ i32, i32 }* [[X_ADDR]] to i8*), i8* [[EXPECTED]], i8* [[DESIRED]], i32 0, i32 0)
// CHECK: br i1 [[SUCCESS_FAIL]], label %[[EXIT:.+]], label %[[CONT]]
// CHECK: [[EXIT]]
// CHECK: store i32 [[OLD_RE]], i32* getelementptr inbounds ({ i32, i32 }, { i32, i32 }* @{{.+}}, i32 0, i32 0),
// CHECK: store i32 [[OLD_IM]], i32* getelementptr inbounds ({ i32, i32 }, { i32, i32 }* @{{.+}}, i32 0, i32 1),
#pragma omp atomic capture
  {civ = cix; cix = lv + cix;}
// CHECK: [[ULV:%.+]] = load i64, i64* @{{.+}},
// CHECK: [[EXPR:%.+]] = uitofp i64 [[ULV]] to float
// CHECK: [[X:%.+]] = load atomic i32, i32*  bitcast (float* [[X_ADDR:@.+]] to i32*) monotonic
// CHECK: br label %[[CONT:.+]]
// CHECK: [[CONT]]
// CHECK: [[EXPECTED:%.+]] = phi i32 [ [[X]], %{{.+}} ], [ [[OLD_X:%.+]], %[[CONT]] ]
// CHECK: [[TEMP_I:%.+]] = bitcast float* [[TEMP:%.+]] to i32*
// CHECK: [[OLD:%.+]] = bitcast i32 [[EXPECTED]] to float
// CHECK: [[MUL:%.+]] = fmul float [[OLD]], [[EXPR]]
// CHECK: store float [[MUL]], float* [[TEMP]],
// CHECK: [[DESIRED:%.+]] = load i32, i32* [[TEMP_I]],
// CHECK: [[RES:%.+]] = cmpxchg i32* bitcast (float* [[X_ADDR]] to i32*), i32 [[EXPECTED]], i32 [[DESIRED]] monotonic monotonic
// CHECK: [[OLD_X:%.+]] = extractvalue { i32, i1 } [[RES]], 0
// CHECK: [[SUCCESS_FAIL:%.+]] = extractvalue { i32, i1 } [[RES]], 1
// CHECK: br i1 [[SUCCESS_FAIL]], label %[[EXIT:.+]], label %[[CONT]]
// CHECK: [[EXIT]]
// CHECK: store float [[MUL]], float* @{{.+}},
#pragma omp atomic capture
  {fx = fx * ulv; fv = fx;}
// CHECK: [[LLV:%.+]] = load i64, i64* @{{.+}},
// CHECK: [[EXPR:%.+]] = sitofp i64 [[LLV]] to double
// CHECK: [[X:%.+]] = load atomic i64, i64*  bitcast (double* [[X_ADDR:@.+]] to i64*) monotonic
// CHECK: br label %[[CONT:.+]]
// CHECK: [[CONT]]
// CHECK: [[EXPECTED:%.+]] = phi i64 [ [[X]], %{{.+}} ], [ [[OLD_X:%.+]], %[[CONT]] ]
// CHECK: [[TEMP_I:%.+]] = bitcast double* [[TEMP:%.+]] to i64*
// CHECK: [[OLD:%.+]] = bitcast i64 [[EXPECTED]] to double
// CHECK: [[DIV:%.+]] = fdiv double [[OLD]], [[EXPR]]
// CHECK: store double [[DIV]], double* [[TEMP]],
// CHECK: [[DESIRED:%.+]] = load i64, i64* [[TEMP_I]],
// CHECK: [[RES:%.+]] = cmpxchg i64* bitcast (double* [[X_ADDR]] to i64*), i64 [[EXPECTED]], i64 [[DESIRED]] monotonic monotonic
// CHECK: [[OLD_X:%.+]] = extractvalue { i64, i1 } [[RES]], 0
// CHECK: [[SUCCESS_FAIL:%.+]] = extractvalue { i64, i1 } [[RES]], 1
// CHECK: br i1 [[SUCCESS_FAIL]], label %[[EXIT:.+]], label %[[CONT]]
// CHECK: [[EXIT]]
// CHECK: store double [[DIV]], double* @{{.+}},
#pragma omp atomic capture
  dv = dx /= llv;
// CHECK: [[ULLV:%.+]] = load i64, i64* @{{.+}},
// CHECK: [[EXPR:%.+]] = uitofp i64 [[ULLV]] to x86_fp80
// CHECK: [[X:%.+]] = load atomic i128, i128*  bitcast (x86_fp80* [[X_ADDR:@.+]] to i128*) monotonic
// CHECK: br label %[[CONT:.+]]
// CHECK: [[CONT]]
// CHECK: [[EXPECTED:%.+]] = phi i128 [ [[X]], %{{.+}} ], [ [[OLD_X:%.+]], %[[CONT]] ]
// CHECK: [[TEMP_I1:%.+]] = bitcast x86_fp80* [[TEMP1:%.+]] to i128*
// CHECK: store i128 [[EXPECTED]], i128* [[TEMP_I1]],
// CHECK: [[TEMP_I:%.+]] = bitcast x86_fp80* [[TEMP:%.+]] to i128*
// CHECK: store i128 [[EXPECTED]], i128* [[TEMP_I]],
// CHECK: [[OLD:%.+]] = load x86_fp80, x86_fp80* [[TEMP]],
// CHECK: [[SUB:%.+]] = fsub x86_fp80 [[OLD]], [[EXPR]]
// CHECK: store x86_fp80 [[SUB]], x86_fp80* [[TEMP1]]
// CHECK: [[DESIRED:%.+]] = load i128, i128* [[TEMP_I1]]
// CHECK: [[RES:%.+]] = cmpxchg i128* bitcast (x86_fp80* [[X_ADDR]] to i128*), i128 [[EXPECTED]], i128 [[DESIRED]] monotonic monotonic
// CHECK: [[OLD_X:%.+]] = extractvalue { i128, i1 } [[RES]], 0
// CHECK: [[SUCCESS_FAIL:%.+]] = extractvalue { i128, i1 } [[RES]], 1
// CHECK: br i1 [[SUCCESS_FAIL]], label %[[EXIT:.+]], label %[[CONT]]
// CHECK: [[EXIT]]
// CHECK: store x86_fp80 [[OLD]], x86_fp80* @{{.+}},
#pragma omp atomic capture
  {ldv = ldx; ldx -= ullv;}
// CHECK: [[EXPR:%.+]] = load float, float* @{{.+}},
// CHECK: [[BITCAST:%.+]] = bitcast { i32, i32 }* [[EXPECTED_ADDR:%.+]] to i8*
// CHECK: call void @__atomic_load(i64 8, i8* bitcast ({ i32, i32 }* [[X_ADDR:@.+]] to i8*), i8* [[BITCAST]], i32 0)
// CHECK: br label %[[CONT:.+]]
// CHECK: [[CONT]]
// CHECK: [[X_RE_ADDR:%.+]] = getelementptr inbounds { i32, i32 }, { i32, i32 }* [[EXPECTED_ADDR]], i32 0, i32 0
// CHECK: [[X_RE:%.+]] = load i32, i32* [[X_RE_ADDR]]
// CHECK: [[X_IM_ADDR:%.+]] = getelementptr inbounds { i32, i32 }, { i32, i32 }* [[EXPECTED_ADDR]], i32 0, i32 1
// CHECK: [[X_IM:%.+]] = load i32, i32* [[X_IM_ADDR]]
// <Skip checks for complex calculations>
// CHECK: [[X_RE_ADDR:%.+]] = getelementptr inbounds { i32, i32 }, { i32, i32 }* [[DESIRED_ADDR:%.+]], i32 0, i32 0
// CHECK: [[X_IM_ADDR:%.+]] = getelementptr inbounds { i32, i32 }, { i32, i32 }* [[DESIRED_ADDR]], i32 0, i32 1
// CHECK: store i32 [[NEW_RE:%.+]], i32* [[X_RE_ADDR]]
// CHECK: store i32 [[NEW_IM:%.+]], i32* [[X_IM_ADDR]]
// CHECK: [[EXPECTED:%.+]] = bitcast { i32, i32 }* [[EXPECTED_ADDR]] to i8*
// CHECK: [[DESIRED:%.+]] = bitcast { i32, i32 }* [[DESIRED_ADDR]] to i8*
// CHECK: [[SUCCESS_FAIL:%.+]] = call zeroext i1 @__atomic_compare_exchange(i64 8, i8* bitcast ({ i32, i32 }* [[X_ADDR]] to i8*), i8* [[EXPECTED]], i8* [[DESIRED]], i32 0, i32 0)
// CHECK: br i1 [[SUCCESS_FAIL]], label %[[EXIT:.+]], label %[[CONT]]
// CHECK: [[EXIT]]
// CHECK: store i32 [[NEW_RE]], i32* getelementptr inbounds ({ i32, i32 }, { i32, i32 }* @{{.+}}, i32 0, i32 0),
// CHECK: store i32 [[NEW_IM]], i32* getelementptr inbounds ({ i32, i32 }, { i32, i32 }* @{{.+}}, i32 0, i32 1),
#pragma omp atomic capture
  {cix = fv / cix; civ = cix;}
// CHECK: [[EXPR:%.+]] = load double, double* @{{.+}},
// CHECK: [[X:%.+]] = load atomic i16, i16* [[X_ADDR:@.+]] monotonic
// CHECK: br label %[[CONT:.+]]
// CHECK: [[CONT]]
// CHECK: [[EXPECTED:%.+]] = phi i16 [ [[X]], %{{.+}} ], [ [[OLD_X:%.+]], %[[CONT]] ]
// CHECK: [[CONV:%.+]] = sext i16 [[EXPECTED]] to i32
// CHECK: [[X_RVAL:%.+]] = sitofp i32 [[CONV]] to double
// CHECK: [[ADD:%.+]] = fadd double [[X_RVAL]], [[EXPR]]
// CHECK: [[NEW:%.+]] = fptosi double [[ADD]] to i16
// CHECK: store i16 [[NEW]], i16* [[TEMP:%.+]],
// CHECK: [[DESIRED:%.+]] = load i16, i16* [[TEMP]],
// CHECK: [[RES:%.+]] = cmpxchg i16* [[X_ADDR]], i16 [[EXPECTED]], i16 [[DESIRED]] monotonic monotonic
// CHECK: [[OLD_X]] = extractvalue { i16, i1 } [[RES]], 0
// CHECK: [[SUCCESS_FAIL:%.+]] = extractvalue { i16, i1 } [[RES]], 1
// CHECK: br i1 [[SUCCESS_FAIL]], label %[[EXIT:.+]], label %[[CONT]]
// CHECK: [[EXIT]]
// CHECK: store i16 [[NEW]], i16* @{{.+}},
#pragma omp atomic capture
  sv = sx = sx + dv;
// CHECK: [[EXPR:%.+]] = load x86_fp80, x86_fp80* @{{.+}},
// CHECK: [[XI8:%.+]] = load atomic i8, i8* [[X_ADDR:@.+]] monotonic
// CHECK: br label %[[CONT:.+]]
// CHECK: [[CONT]]
// CHECK: [[EXPECTED:%.+]] = phi i8 [ [[XI8]], %{{.+}} ], [ [[OLD_X:%.+]], %[[CONT]] ]
// CHECK: [[BOOL_EXPECTED:%.+]] = trunc i8 [[EXPECTED]] to i1
// CHECK: [[CONV:%.+]] = zext i1 [[BOOL_EXPECTED]] to i32
// CHECK: [[X_RVAL:%.+]] = sitofp i32 [[CONV]] to x86_fp80
// CHECK: [[MUL:%.+]] = fmul x86_fp80 [[EXPR]], [[X_RVAL]]
// CHECK: [[BOOL_DESIRED:%.+]] = fcmp une x86_fp80 [[MUL]], 0xK00000000000000000000
// CHECK: [[DESIRED:%.+]] = zext i1 [[BOOL_DESIRED]] to i8
// CHECK: store i8 [[DESIRED]], i8* [[TEMP:%.+]],
// CHECK: [[DESIRED:%.+]] = load i8, i8* [[TEMP]],
// CHECK: [[RES:%.+]] = cmpxchg i8* [[X_ADDR]], i8 [[EXPECTED]], i8 [[DESIRED]] monotonic monotonic
// CHECK: [[OLD_X:%.+]] = extractvalue { i8, i1 } [[RES]], 0
// CHECK: [[SUCCESS_FAIL:%.+]] = extractvalue { i8, i1 } [[RES]], 1
// CHECK: br i1 [[SUCCESS_FAIL]], label %[[EXIT:.+]], label %[[CONT]]
// CHECK: [[EXIT]]
// CHECK: [[EXPECTED_I8:%.+]] = zext i1 [[BOOL_EXPECTED]] to i8
// CHECK: store i8 [[EXPECTED_I8]], i8* @{{.+}},
#pragma omp atomic capture
  {bv = bx; bx = ldv * bx;}
// CHECK: [[EXPR_RE:%.+]] = load i32, i32* getelementptr inbounds ({ i32, i32 }, { i32, i32 }* [[CIV_ADDR:@.+]], i32 0, i32 0),
// CHECK: [[EXPR_IM:%.+]] = load i32, i32* getelementptr inbounds ({ i32, i32 }, { i32, i32 }* [[CIV_ADDR]], i32 0, i32 1),
// CHECK: [[XI8:%.+]] = load atomic i8, i8* [[X_ADDR:@.+]] monotonic
// CHECK: br label %[[CONT:.+]]
// CHECK: [[CONT]]
// CHECK: [[EXPECTED:%.+]] = phi i8 [ [[XI8]], %{{.+}} ], [ [[OLD_X:%.+]], %[[CONT]] ]
// CHECK: [[BOOL_EXPECTED:%.+]] = trunc i8 [[EXPECTED]] to i1
// CHECK: [[X_RVAL:%.+]] = zext i1 [[BOOL_EXPECTED]] to i32
// CHECK: [[SUB_RE:%.+]] = sub i32 [[EXPR_RE:%.+]], [[X_RVAL]]
// CHECK: [[SUB_IM:%.+]] = sub i32 [[EXPR_IM:%.+]], 0
// CHECK: icmp ne i32 [[SUB_RE]], 0
// CHECK: icmp ne i32 [[SUB_IM]], 0
// CHECK: [[BOOL_DESIRED:%.+]] = or i1
// CHECK: [[DESIRED:%.+]] = zext i1 [[BOOL_DESIRED]] to i8
// CHECK: store i8 [[DESIRED]], i8* [[TEMP:%.+]],
// CHECK: [[DESIRED:%.+]] = load i8, i8* [[TEMP]],
// CHECK: [[RES:%.+]] = cmpxchg i8* [[X_ADDR]], i8 [[EXPECTED]], i8 [[DESIRED]] monotonic monotonic
// CHECK: [[OLD_X:%.+]] = extractvalue { i8, i1 } [[RES]], 0
// CHECK: [[SUCCESS_FAIL:%.+]] = extractvalue { i8, i1 } [[RES]], 1
// CHECK: br i1 [[SUCCESS_FAIL]], label %[[EXIT:.+]], label %[[CONT]]
// CHECK: [[EXIT]]
// CHECK: [[DESIRED_I8:%.+]] = zext i1 [[BOOL_DESIRED]] to i8
// CHECK: store i8 [[DESIRED_I8]], i8* @{{.+}},
#pragma omp atomic capture
  {bx = civ - bx; bv = bx;}
// CHECK: [[EXPR_RE:%.+]] = load float, float* getelementptr inbounds ({ float, float }, { float, float }* @{{.+}}, i32 0, i32 0)
// CHECK: [[EXPR_IM:%.+]] = load float, float* getelementptr inbounds ({ float, float }, { float, float }* @{{.+}}, i32 0, i32 1)
// CHECK: [[X:%.+]] = load atomic i16, i16* [[X_ADDR:@.+]] monotonic
// CHECK: br label %[[CONT:.+]]
// CHECK: [[CONT]]
// CHECK: [[EXPECTED:%.+]] = phi i16 [ [[X]], %{{.+}} ], [ [[OLD_X:%.+]], %[[CONT]] ]
// CHECK: [[CONV:%.+]] = zext i16 [[EXPECTED]] to i32
// CHECK: [[X_RVAL:%.+]] = sitofp i32 [[CONV]] to float
// <Skip checks for complex calculations>
// CHECK: [[X_RE_ADDR:%.+]] = getelementptr inbounds { float, float }, { float, float }* [[TEMP:%.+]], i32 0, i32 0
// CHECK: [[X_RE:%.+]] = load float, float* [[X_RE_ADDR]]
// CHECK: [[X_IM_ADDR:%.+]] = getelementptr inbounds { float, float }, { float, float }* [[TEMP]], i32 0, i32 1
// CHECK: [[X_IM:%.+]] = load float, float* [[X_IM_ADDR]]
// CHECK: [[NEW:%.+]] = fptoui float [[X_RE]] to i16
// CHECK: store i16 [[NEW]], i16* [[TEMP:%.+]],
// CHECK: [[DESIRED:%.+]] = load i16, i16* [[TEMP]],
// CHECK: [[RES:%.+]] = cmpxchg i16* [[X_ADDR]], i16 [[EXPECTED]], i16 [[DESIRED]] monotonic monotonic
// CHECK: [[OLD_X]] = extractvalue { i16, i1 } [[RES]], 0
// CHECK: [[SUCCESS_FAIL:%.+]] = extractvalue { i16, i1 } [[RES]], 1
// CHECK: br i1 [[SUCCESS_FAIL]], label %[[EXIT:.+]], label %[[CONT]]
// CHECK: [[EXIT]]
// CHECK: store i16 [[NEW]], i16* @{{.+}},
#pragma omp atomic capture
  usv = usx /= cfv;
// CHECK: [[EXPR_RE:%.+]] = load double, double* getelementptr inbounds ({ double, double }, { double, double }* @{{.+}}, i32 0, i32 0)
// CHECK: [[EXPR_IM:%.+]] = load double, double* getelementptr inbounds ({ double, double }, { double, double }* @{{.+}}, i32 0, i32 1)
// CHECK: [[X:%.+]] = load atomic i64, i64* [[X_ADDR:@.+]] monotonic
// CHECK: br label %[[CONT:.+]]
// CHECK: [[CONT]]
// CHECK: [[EXPECTED:%.+]] = phi i64 [ [[X]], %{{.+}} ], [ [[OLD_X:%.+]], %[[CONT]] ]
// CHECK: [[X_RVAL:%.+]] = sitofp i64 [[EXPECTED]] to double
// CHECK: [[ADD_RE:%.+]] = fadd double [[X_RVAL]], [[EXPR_RE]]
// CHECK: [[ADD_IM:%.+]] = fadd double 0.000000e+00, [[EXPR_IM]]
// CHECK: [[DESIRED:%.+]] = fptosi double [[ADD_RE]] to i64
// CHECK: store i64 [[DESIRED]], i64* [[TEMP:%.+]],
// CHECK: [[DESIRED:%.+]] = load i64, i64* [[TEMP]],
// CHECK: [[RES:%.+]] = cmpxchg i64* [[X_ADDR]], i64 [[EXPECTED]], i64 [[DESIRED]] monotonic monotonic
// CHECK: [[OLD_X]] = extractvalue { i64, i1 } [[RES]], 0
// CHECK: [[SUCCESS_FAIL:%.+]] = extractvalue { i64, i1 } [[RES]], 1
// CHECK: br i1 [[SUCCESS_FAIL]], label %[[EXIT:.+]], label %[[CONT]]
// CHECK: [[EXIT]]
// CHECK: store i64 [[EXPECTED]], i64* @{{.+}},
#pragma omp atomic capture
  {llv = llx; llx += cdv;}
// CHECK: [[IDX:%.+]] = load i16, i16* @{{.+}}
// CHECK: load i8, i8*
// CHECK: [[VEC_ITEM_VAL:%.+]] = zext i1 %{{.+}} to i32
// CHECK: [[I128VAL:%.+]] = load atomic i128, i128* bitcast (<4 x i32>* [[DEST:@.+]] to i128*) monotonic
// CHECK: br label %[[CONT:.+]]
// CHECK: [[CONT]]
// CHECK: [[OLD_I128:%.+]] = phi i128 [ [[I128VAL]], %{{.+}} ], [ [[FAILED_OLD_VAL:%.+]], %[[CONT]] ]
// CHECK: [[TEMP_I:%.+]] = bitcast <4 x i32>* [[TEMP:%.+]] to i128*
// CHECK: store i128 [[OLD_I128]], i128* [[TEMP_I]],
// CHECK: [[LD:%.+]] = bitcast i128 [[OLD_I128]] to <4 x i32>
// CHECK: store <4 x i32> [[LD]], <4 x i32>* [[TEMP1:%.+]],
// CHECK: [[VEC_VAL:%.+]] = load <4 x i32>, <4 x i32>* [[TEMP1]]
// CHECK: [[ITEM:%.+]] = extractelement <4 x i32> [[VEC_VAL]], i16 [[IDX]]
// CHECK: [[OR:%.+]] = or i32 [[ITEM]], [[VEC_ITEM_VAL]]
// CHECK: [[VEC_VAL:%.+]] = load <4 x i32>, <4 x i32>* [[TEMP]]
// CHECK: [[NEW_VEC_VAL:%.+]] = insertelement <4 x i32> [[VEC_VAL]], i32 [[OR]], i16 [[IDX]]
// CHECK: store <4 x i32> [[NEW_VEC_VAL]], <4 x i32>* [[TEMP]]
// CHECK: [[NEW_I128:%.+]] = load i128, i128* [[TEMP_I]],
// CHECK: [[RES:%.+]] = cmpxchg i128* bitcast (<4 x i32>* [[DEST]] to i128*), i128 [[OLD_I128]], i128 [[NEW_I128]] monotonic monotonic
// CHECK: [[FAILED_OLD_VAL:%.+]] = extractvalue { i128, i1 } [[RES]], 0
// CHECK: [[FAIL_SUCCESS:%.+]] = extractvalue { i128, i1 } [[RES]], 1
// CHECK: br i1 [[FAIL_SUCCESS]], label %[[EXIT:.+]], label %[[CONT]]
// CHECK: [[EXIT]]
// CHECK: store i32 [[OR]], i32* @{{.+}},
#pragma omp atomic capture
  {int4x[sv] |= bv; iv = int4x[sv];}
// CHECK: [[EXPR:%.+]] = load x86_fp80, x86_fp80* @{{.+}}
// CHECK: [[PREV_VALUE:%.+]] = load atomic i32, i32* bitcast (i8* getelementptr (i8, i8* bitcast (%struct.BitFields* @{{.+}} to i8*), i64 4) to i32*) monotonic
// CHECK: br label %[[CONT:.+]]
// CHECK: [[CONT]]
// CHECK: [[OLD_BF_VALUE:%.+]] = phi i32 [ [[PREV_VALUE]], %[[EXIT]] ], [ [[FAILED_OLD_VAL:%.+]], %[[CONT]] ]
// CHECK: store i32 [[OLD_BF_VALUE]], i32* [[TEMP1:%.+]],
// CHECK: store i32 [[OLD_BF_VALUE]], i32* [[TEMP:%.+]],
// CHECK: [[A_LD:%.+]] = load i32, i32* [[TEMP]],
// CHECK: [[A_SHL:%.+]] = shl i32 [[A_LD]], 1
// CHECK: [[A_ASHR:%.+]] = ashr i32 [[A_SHL]], 1
// CHECK: [[X_RVAL:%.+]] = sitofp i32 [[A_ASHR]] to x86_fp80
// CHECK: [[SUB:%.+]] = fsub x86_fp80 [[X_RVAL]], [[EXPR]]
// CHECK: [[CONV:%.+]] = fptosi x86_fp80 [[SUB]] to i32
// CHECK: [[NEW_VAL:%.+]] = load i32, i32* [[TEMP1]],
// CHECK: [[BF_VALUE:%.+]] = and i32 [[CONV]], 2147483647
// CHECK: [[BF_CLEAR:%.+]] = and i32 [[NEW_VAL]], -2147483648
// CHECK: [[BF_SET:%.+]] = or i32 [[BF_CLEAR]], [[BF_VALUE]]
// CHECK: store i32 [[BF_SET]], i32* [[TEMP1]],
// CHECK: [[NEW_BF_VALUE:%.+]] = load i32, i32* [[TEMP1]],
// CHECK: [[RES:%.+]] = cmpxchg i32* bitcast (i8* getelementptr (i8, i8* bitcast (%struct.BitFields* @{{.+}} to i8*), i64 4) to i32*), i32 [[OLD_BF_VALUE]], i32 [[NEW_BF_VALUE]] monotonic monotonic
// CHECK: [[FAILED_OLD_VAL]] = extractvalue { i32, i1 } [[RES]], 0
// CHECK: [[FAIL_SUCCESS:%.+]] = extractvalue { i32, i1 } [[RES]], 1
// CHECK: br i1 [[FAIL_SUCCESS]], label %[[EXIT:.+]], label %[[CONT]]
// CHECK: [[EXIT]]
// CHECK: store i32 [[CONV]], i32* @{{.+}},
#pragma omp atomic capture
  iv = bfx.a = bfx.a - ldv;
// CHECK: [[EXPR:%.+]] = load x86_fp80, x86_fp80* @{{.+}}
// CHECK: [[BITCAST:%.+]] = bitcast i32* [[LDTEMP:%.+]] to i8*
// CHECK: call void @__atomic_load(i64 4, i8* getelementptr (i8, i8* bitcast (%struct.BitFields_packed* @{{.+}} to i8*), i64 4), i8* [[BITCAST]], i32 0)
// CHECK: br label %[[CONT:.+]]
// CHECK: [[CONT]]
// CHECK: [[OLD:%.+]] = load i32, i32* [[LDTEMP]],
// CHECK: store i32 [[OLD]], i32* [[TEMP1:%.+]],
// CHECK: [[OLD:%.+]] = load i32, i32* [[LDTEMP]],
// CHECK: store i32 [[OLD]], i32* [[TEMP:%.+]],
// CHECK: [[A_LD:%.+]] = load i32, i32* [[TEMP]],
// CHECK: [[A_SHL:%.+]] = shl i32 [[A_LD]], 1
// CHECK: [[A_ASHR:%.+]] = ashr i32 [[A_SHL]], 1
// CHECK: [[X_RVAL:%.+]] = sitofp i32 [[A_ASHR]] to x86_fp80
// CHECK: [[MUL:%.+]] = fmul x86_fp80 [[X_RVAL]], [[EXPR]]
// CHECK: [[CONV:%.+]] = fptosi x86_fp80 [[MUL]] to i32
// CHECK: [[NEW_VAL:%.+]] = load i32, i32* [[TEMP1]],
// CHECK: [[BF_VALUE:%.+]] = and i32 [[CONV]], 2147483647
// CHECK: [[BF_CLEAR:%.+]] = and i32 [[NEW_VAL]], -2147483648
// CHECK: or i32 [[BF_CLEAR]], [[BF_VALUE]]
// CHECK: store i32 %{{.+}}, i32* [[TEMP1]]
// CHECK: [[BITCAST_TEMP_OLD_BF_ADDR:%.+]] = bitcast i32* [[LDTEMP]] to i8*
// CHECK: [[BITCAST_TEMP_NEW_BF_ADDR:%.+]] = bitcast i32* [[TEMP1]] to i8*
// CHECK: [[FAIL_SUCCESS:%.+]] = call zeroext i1 @__atomic_compare_exchange(i64 4, i8* getelementptr (i8, i8* bitcast (%struct.BitFields_packed* @{{.+}} to i8*), i64 4), i8* [[BITCAST_TEMP_OLD_BF_ADDR]], i8* [[BITCAST_TEMP_NEW_BF_ADDR]], i32 0, i32 0)
// CHECK: br i1 [[FAIL_SUCCESS]], label %[[EXIT:.+]], label %[[CONT]]
// CHECK: [[EXIT]]
// CHECK: store i32 [[A_ASHR]], i32* @{{.+}},
#pragma omp atomic capture
  {iv = bfx_packed.a; bfx_packed.a *= ldv;}
// CHECK: [[EXPR:%.+]] = load x86_fp80, x86_fp80* @{{.+}}
// CHECK: [[PREV_VALUE:%.+]] = load atomic i32, i32* getelementptr inbounds (%struct.BitFields2, %struct.BitFields2* @{{.+}}, i32 0, i32 0) monotonic
// CHECK: br label %[[CONT:.+]]
// CHECK: [[CONT]]
// CHECK: [[OLD_BF_VALUE:%.+]] = phi i32 [ [[PREV_VALUE]], %[[EXIT]] ], [ [[FAILED_OLD_VAL:%.+]], %[[CONT]] ]
// CHECK: store i32 [[OLD_BF_VALUE]], i32* [[TEMP1:%.+]],
// CHECK: store i32 [[OLD_BF_VALUE]], i32* [[TEMP:%.+]],
// CHECK: [[A_LD:%.+]] = load i32, i32* [[TEMP]],
// CHECK: [[A_ASHR:%.+]] = ashr i32 [[A_LD]], 31
// CHECK: [[X_RVAL:%.+]] = sitofp i32 [[A_ASHR]] to x86_fp80
// CHECK: [[SUB:%.+]] = fsub x86_fp80 [[X_RVAL]], [[EXPR]]
// CHECK: [[CONV:%.+]] = fptosi x86_fp80 [[SUB]] to i32
// CHECK: [[NEW_VAL:%.+]] = load i32, i32* [[TEMP1]],
// CHECK: [[BF_AND:%.+]] = and i32 [[CONV]], 1
// CHECK: [[BF_VALUE:%.+]] = shl i32 [[BF_AND]], 31
// CHECK: [[BF_CLEAR:%.+]] = and i32 [[NEW_VAL]], 2147483647
// CHECK: or i32 [[BF_CLEAR]], [[BF_VALUE]]
// CHECK: store i32 %{{.+}}, i32* [[TEMP1]]
// CHECK: [[NEW_BF_VALUE:%.+]] = load i32, i32* [[TEMP1]]
// CHECK: [[RES:%.+]] = cmpxchg i32* getelementptr inbounds (%struct.BitFields2, %struct.BitFields2* @{{.+}}, i32 0, i32 0), i32 [[OLD_BF_VALUE]], i32 [[NEW_BF_VALUE]] monotonic monotonic
// CHECK: [[FAILED_OLD_VAL]] = extractvalue { i32, i1 } [[RES]], 0
// CHECK: [[FAIL_SUCCESS:%.+]] = extractvalue { i32, i1 } [[RES]], 1
// CHECK: br i1 [[FAIL_SUCCESS]], label %[[EXIT:.+]], label %[[CONT]]
// CHECK: [[EXIT]]
// CHECK: store i32 [[CONV]], i32* @{{.+}},
#pragma omp atomic capture
  {bfx2.a -= ldv; iv = bfx2.a;}
// CHECK: [[EXPR:%.+]] = load x86_fp80, x86_fp80* @{{.+}}
// CHECK: [[PREV_VALUE:%.+]] = load atomic i8, i8* getelementptr (i8, i8* bitcast (%struct.BitFields2_packed* @{{.+}} to i8*), i64 3) monotonic
// CHECK: br label %[[CONT:.+]]
// CHECK: [[CONT]]
// CHECK: [[OLD_BF_VALUE:%.+]] = phi i8 [ [[PREV_VALUE]], %[[EXIT]] ], [ [[FAILED_OLD_VAL:%.+]], %[[CONT]] ]
// CHECK: [[BITCAST_NEW:%.+]] = bitcast i32* %{{.+}} to i8*
// CHECK: store i8 [[OLD_BF_VALUE]], i8* [[BITCAST_NEW]],
// CHECK: [[BITCAST:%.+]] = bitcast i32* %{{.+}} to i8*
// CHECK: store i8 [[OLD_BF_VALUE]], i8* [[BITCAST]],
// CHECK: [[A_LD:%.+]] = load i8, i8* [[BITCAST]],
// CHECK: [[A_ASHR:%.+]] = ashr i8 [[A_LD]], 7
// CHECK: [[CAST:%.+]] = sext i8 [[A_ASHR]] to i32
// CHECK: [[X_RVAL:%.+]] = sitofp i32 [[CAST]] to x86_fp80
// CHECK: [[DIV:%.+]] = fdiv x86_fp80 [[EXPR]], [[X_RVAL]]
// CHECK: [[NEW_VAL:%.+]] = fptosi x86_fp80 [[DIV]] to i32
// CHECK: [[TRUNC:%.+]] = trunc i32 [[NEW_VAL]] to i8
// CHECK: [[BF_LD:%.+]] = load i8, i8* [[BITCAST_NEW]],
// CHECK: [[BF_AND:%.+]] = and i8 [[TRUNC]], 1
// CHECK: [[BF_VALUE:%.+]] = shl i8 [[BF_AND]], 7
// CHECK: [[BF_CLEAR:%.+]] = and i8 %{{.+}}, 127
// CHECK: or i8 [[BF_CLEAR]], [[BF_VALUE]]
// CHECK: store i8 %{{.+}}, i8* [[BITCAST_NEW]]
// CHECK: [[NEW_BF_VALUE:%.+]] = load i8, i8* [[BITCAST_NEW]]
// CHECK: [[RES:%.+]] = cmpxchg i8* getelementptr (i8, i8* bitcast (%struct.BitFields2_packed* @{{.+}} to i8*), i64 3), i8 [[OLD_BF_VALUE]], i8 [[NEW_BF_VALUE]] monotonic monotonic
// CHECK: [[FAILED_OLD_VAL]] = extractvalue { i8, i1 } [[RES]], 0
// CHECK: [[FAIL_SUCCESS:%.+]] = extractvalue { i8, i1 } [[RES]], 1
// CHECK: br i1 [[FAIL_SUCCESS]], label %[[EXIT:.+]], label %[[CONT]]
// CHECK: [[EXIT]]
// CHECK: store i32 [[NEW_VAL]], i32* @{{.+}},
#pragma omp atomic capture
  iv = bfx2_packed.a = ldv / bfx2_packed.a;
// CHECK: [[EXPR:%.+]] = load x86_fp80, x86_fp80* @{{.+}}
// CHECK: [[PREV_VALUE:%.+]] = load atomic i32, i32* getelementptr inbounds (%struct.BitFields3, %struct.BitFields3* @{{.+}}, i32 0, i32 0) monotonic
// CHECK: br label %[[CONT:.+]]
// CHECK: [[CONT]]
// CHECK: [[OLD_BF_VALUE:%.+]] = phi i32 [ [[PREV_VALUE]], %[[EXIT]] ], [ [[FAILED_OLD_VAL:%.+]], %[[CONT]] ]
// CHECK: store i32 [[OLD_BF_VALUE]], i32* [[TEMP1:%.+]],
// CHECK: store i32 [[OLD_BF_VALUE]], i32* [[TEMP:%.+]],
// CHECK: [[A_LD:%.+]] = load i32, i32* [[TEMP]],
// CHECK: [[A_SHL:%.+]] = shl i32 [[A_LD]], 7
// CHECK: [[A_ASHR:%.+]] = ashr i32 [[A_SHL]], 18
// CHECK: [[X_RVAL:%.+]] = sitofp i32 [[A_ASHR]] to x86_fp80
// CHECK: [[DIV:%.+]] = fdiv x86_fp80 [[X_RVAL]], [[EXPR]]
// CHECK: [[NEW_VAL:%.+]] = fptosi x86_fp80 [[DIV]] to i32
// CHECK: [[BF_LD:%.+]] = load i32, i32* [[TEMP1]],
// CHECK: [[BF_AND:%.+]] = and i32 [[NEW_VAL]], 16383
// CHECK: [[BF_VALUE:%.+]] = shl i32 [[BF_AND]], 11
// CHECK: [[BF_CLEAR:%.+]] = and i32 %{{.+}}, -33552385
// CHECK: or i32 [[BF_CLEAR]], [[BF_VALUE]]
// CHECK: store i32 %{{.+}}, i32* [[TEMP1]]
// CHECK: [[NEW_BF_VALUE:%.+]] = load i32, i32* [[TEMP1]]
// CHECK: [[RES:%.+]] = cmpxchg i32* getelementptr inbounds (%struct.BitFields3, %struct.BitFields3* @{{.+}}, i32 0, i32 0), i32 [[OLD_BF_VALUE]], i32 [[NEW_BF_VALUE]] monotonic monotonic
// CHECK: [[FAILED_OLD_VAL]] = extractvalue { i32, i1 } [[RES]], 0
// CHECK: [[FAIL_SUCCESS:%.+]] = extractvalue { i32, i1 } [[RES]], 1
// CHECK: br i1 [[FAIL_SUCCESS]], label %[[EXIT:.+]], label %[[CONT]]
// CHECK: [[EXIT]]
// CHECK: store i32 [[A_ASHR]], i32* @{{.+}},
#pragma omp atomic capture
  {iv = bfx3.a; bfx3.a /= ldv;}
// CHECK: [[EXPR:%.+]] = load x86_fp80, x86_fp80* @{{.+}}
// CHECK: [[LDTEMP:%.+]] = bitcast i32* %{{.+}} to i24*
// CHECK: [[BITCAST:%.+]] = bitcast i24* [[LDTEMP]] to i8*
// CHECK: call void @__atomic_load(i64 3, i8* getelementptr (i8, i8* bitcast (%struct.BitFields3_packed* @{{.+}} to i8*), i64 1), i8* [[BITCAST]], i32 0)
// CHECK: br label %[[CONT:.+]]
// CHECK: [[CONT]]
// CHECK: [[OLD:%.+]] = load i24, i24* [[LDTEMP]],
// CHECK: store i24 [[OLD]], i24* [[BITCAST2:%.+]],
// CHECK: [[OLD:%.+]] = load i24, i24* [[LDTEMP]],
// CHECK: store i24 [[OLD]], i24* [[BITCAST1:%.+]],
// CHECK: [[A_LD:%.+]] = load i24, i24* [[BITCAST1]],
// CHECK: [[A_SHL:%.+]] = shl i24 [[A_LD]], 7
// CHECK: [[A_ASHR:%.+]] = ashr i24 [[A_SHL]], 10
// CHECK: [[CAST:%.+]] = sext i24 [[A_ASHR]] to i32
// CHECK: [[X_RVAL:%.+]] = sitofp i32 [[CAST]] to x86_fp80
// CHECK: [[ADD:%.+]] = fadd x86_fp80 [[X_RVAL]], [[EXPR]]
// CHECK: [[NEW_VAL:%.+]] = fptosi x86_fp80 [[ADD]] to i32
// CHECK: [[TRUNC:%.+]] = trunc i32 [[NEW_VAL]] to i24
// CHECK: [[BF_LD:%.+]] = load i24, i24* [[BITCAST2]],
// CHECK: [[BF_AND:%.+]] = and i24 [[TRUNC]], 16383
// CHECK: [[BF_VALUE:%.+]] = shl i24 [[BF_AND]], 3
// CHECK: [[BF_CLEAR:%.+]] = and i24 [[BF_LD]], -131065
// CHECK: or i24 [[BF_CLEAR]], [[BF_VALUE]]
// CHECK: store i24 %{{.+}}, i24* [[BITCAST2]]
// CHECK: [[BITCAST_TEMP_OLD_BF_ADDR:%.+]] = bitcast i24* [[LDTEMP]] to i8*
// CHECK: [[BITCAST_TEMP_NEW_BF_ADDR:%.+]] = bitcast i24* [[BITCAST2]] to i8*
// CHECK: [[FAIL_SUCCESS:%.+]] = call zeroext i1 @__atomic_compare_exchange(i64 3, i8* getelementptr (i8, i8* bitcast (%struct.BitFields3_packed* @{{.+}} to i8*), i64 1), i8* [[BITCAST_TEMP_OLD_BF_ADDR]], i8* [[BITCAST_TEMP_NEW_BF_ADDR]], i32 0, i32 0)
// CHECK: br i1 [[FAIL_SUCCESS]], label %[[EXIT:.+]], label %[[CONT]]
// CHECK: [[EXIT]]
// CHECK: store i32 [[NEW_VAL]], i32* @{{.+}},
#pragma omp atomic capture
  {bfx3_packed.a += ldv; iv = bfx3_packed.a;}
// CHECK: [[EXPR:%.+]] = load x86_fp80, x86_fp80* @{{.+}}
// CHECK: [[PREV_VALUE:%.+]] = load atomic i64, i64* bitcast (%struct.BitFields4* @{{.+}} to i64*) monotonic
// CHECK: br label %[[CONT:.+]]
// CHECK: [[CONT]]
// CHECK: [[OLD_BF_VALUE:%.+]] = phi i64 [ [[PREV_VALUE]], %[[EXIT]] ], [ [[FAILED_OLD_VAL:%.+]], %[[CONT]] ]
// CHECK: store i64 [[OLD_BF_VALUE]], i64* [[TEMP1:%.+]],
// CHECK: store i64 [[OLD_BF_VALUE]], i64* [[TEMP:%.+]],
// CHECK: [[A_LD:%.+]] = load i64, i64* [[TEMP]],
// CHECK: [[A_SHL:%.+]] = shl i64 [[A_LD]], 47
// CHECK: [[A_ASHR:%.+]] = ashr i64 [[A_SHL:%.+]], 63
// CHECK: [[A_CAST:%.+]] = trunc i64 [[A_ASHR:%.+]] to i32
// CHECK: [[X_RVAL:%.+]] = sitofp i32 [[CAST:%.+]] to x86_fp80
// CHECK: [[MUL:%.+]] = fmul x86_fp80 [[X_RVAL]], [[EXPR]]
// CHECK: [[NEW_VAL:%.+]] = fptosi x86_fp80 [[MUL]] to i32
// CHECK: [[ZEXT:%.+]] = zext i32 [[NEW_VAL]] to i64
// CHECK: [[BF_LD:%.+]] = load i64, i64* [[TEMP1]],
// CHECK: [[BF_AND:%.+]] = and i64 [[ZEXT]], 1
// CHECK: [[BF_VALUE:%.+]] = shl i64 [[BF_AND]], 16
// CHECK: [[BF_CLEAR:%.+]] = and i64 [[BF_LD]], -65537
// CHECK: or i64 [[BF_CLEAR]], [[BF_VALUE]]
// CHECK: store i64 %{{.+}}, i64* [[TEMP1]]
// CHECK: [[NEW_BF_VALUE:%.+]] = load i64, i64* [[TEMP1]]
// CHECK: [[RES:%.+]] = cmpxchg i64* bitcast (%struct.BitFields4* @{{.+}} to i64*), i64 [[OLD_BF_VALUE]], i64 [[NEW_BF_VALUE]] monotonic monotonic
// CHECK: [[FAILED_OLD_VAL]] = extractvalue { i64, i1 } [[RES]], 0
// CHECK: [[FAIL_SUCCESS:%.+]] = extractvalue { i64, i1 } [[RES]], 1
// CHECK: br i1 [[FAIL_SUCCESS]], label %[[EXIT:.+]], label %[[CONT]]
// CHECK: [[EXIT]]
// CHECK: store i32 [[NEW_VAL]], i32* @{{.+}},
#pragma omp atomic capture
  iv = bfx4.a = bfx4.a * ldv;
// CHECK: [[EXPR:%.+]] = load x86_fp80, x86_fp80* @{{.+}}
// CHECK: [[PREV_VALUE:%.+]] = load atomic i8, i8* getelementptr inbounds (%struct.BitFields4_packed, %struct.BitFields4_packed* @{{.+}}, i32 0, i32 0, i64 2) monotonic
// CHECK: br label %[[CONT:.+]]
// CHECK: [[CONT]]
// CHECK: [[OLD_BF_VALUE:%.+]] = phi i8 [ [[PREV_VALUE]], %{{.+}} ], [ [[FAILED_OLD_VAL:%.+]], %[[CONT]] ]
// CHECK: [[BITCAST1:%.+]] = bitcast i32* %{{.+}} to i8*
// CHECK: store i8 [[OLD_BF_VALUE]], i8* [[BITCAST1]],
// CHECK: [[BITCAST:%.+]] = bitcast i32* %{{.+}} to i8*
// CHECK: store i8 [[OLD_BF_VALUE]], i8* [[BITCAST]],
// CHECK: [[A_LD:%.+]] = load i8, i8* [[BITCAST]],
// CHECK: [[A_SHL:%.+]] = shl i8 [[A_LD]], 7
// CHECK: [[A_ASHR:%.+]] = ashr i8 [[A_SHL:%.+]], 7
// CHECK: [[CAST:%.+]] = sext i8 [[A_ASHR:%.+]] to i32
// CHECK: [[CONV:%.+]] = sitofp i32 [[CAST]] to x86_fp80
// CHECK: [[SUB: %.+]] = fsub x86_fp80 [[CONV]], [[EXPR]]
// CHECK: [[CONV:%.+]] = fptosi x86_fp80 [[SUB:%.+]] to i32
// CHECK: [[NEW_VAL:%.+]] = trunc i32 [[CONV]] to i8
// CHECK: [[BF_LD:%.+]] = load i8, i8* [[BITCAST1]],
// CHECK: [[BF_VALUE:%.+]] = and i8 [[NEW_VAL]], 1
// CHECK: [[BF_CLEAR:%.+]] = and i8 [[BF_LD]], -2
// CHECK: or i8 [[BF_CLEAR]], [[BF_VALUE]]
// CHECK: store i8 %{{.+}}, i8* [[BITCAST1]]
// CHECK: [[NEW_BF_VALUE:%.+]] = load i8, i8* [[BITCAST1]]
// CHECK: [[RES:%.+]] = cmpxchg i8* getelementptr inbounds (%struct.BitFields4_packed, %struct.BitFields4_packed* @{{.+}}, i32 0, i32 0, i64 2), i8 [[OLD_BF_VALUE]], i8 [[NEW_BF_VALUE]] monotonic monotonic
// CHECK: [[FAILED_OLD_VAL]] = extractvalue { i8, i1 } [[RES]], 0
// CHECK: [[FAIL_SUCCESS:%.+]] = extractvalue { i8, i1 } [[RES]], 1
// CHECK: br i1 [[FAIL_SUCCESS]], label %[[EXIT:.+]], label %[[CONT]]
// CHECK: [[EXIT]]
// CHECK: store i32 [[CAST]], i32* @{{.+}},
#pragma omp atomic capture
  {iv = bfx4_packed.a; bfx4_packed.a -= ldv;}
// CHECK: [[EXPR:%.+]] = load x86_fp80, x86_fp80* @{{.+}}
// CHECK: [[PREV_VALUE:%.+]] = load atomic i64, i64* bitcast (%struct.BitFields4* @{{.+}} to i64*) monotonic
// CHECK: br label %[[CONT:.+]]
// CHECK: [[CONT]]
// CHECK: [[OLD_BF_VALUE:%.+]] = phi i64 [ [[PREV_VALUE]], %[[EXIT]] ], [ [[FAILED_OLD_VAL:%.+]], %[[CONT]] ]
// CHECK: store i64 [[OLD_BF_VALUE]], i64* [[TEMP1:%.+]],
// CHECK: store i64 [[OLD_BF_VALUE]], i64* [[TEMP:%.+]],
// CHECK: [[A_LD:%.+]] = load i64, i64* [[TEMP]],
// CHECK: [[A_SHL:%.+]] = shl i64 [[A_LD]], 40
// CHECK: [[A_ASHR:%.+]] = ashr i64 [[A_SHL:%.+]], 57
// CHECK: [[CONV:%.+]] = sitofp i64 [[A_ASHR]] to x86_fp80
// CHECK: [[DIV:%.+]] = fdiv x86_fp80 [[CONV]], [[EXPR]]
// CHECK: [[CONV:%.+]] = fptosi x86_fp80 [[DIV]] to i64
// CHECK: [[BF_LD:%.+]] = load i64, i64* [[TEMP1]],
// CHECK: [[BF_AND:%.+]] = and i64 [[CONV]], 127
// CHECK: [[BF_VALUE:%.+]] = shl i64 [[BF_AND:%.+]], 17
// CHECK: [[BF_CLEAR:%.+]] = and i64 [[BF_LD]], -16646145
// CHECK: [[VAL:%.+]] = or i64 [[BF_CLEAR]], [[BF_VALUE]]
// CHECK: store i64 [[VAL]], i64* [[TEMP1]]
// CHECK: [[NEW_BF_VALUE:%.+]] = load i64, i64* [[TEMP1]]
// CHECK: [[RES:%.+]] = cmpxchg i64* bitcast (%struct.BitFields4* @{{.+}} to i64*), i64 [[OLD_BF_VALUE]], i64 [[NEW_BF_VALUE]] monotonic monotonic
// CHECK: [[FAILED_OLD_VAL]] = extractvalue { i64, i1 } [[RES]], 0
// CHECK: [[FAIL_SUCCESS:%.+]] = extractvalue { i64, i1 } [[RES]], 1
// CHECK: br i1 [[FAIL_SUCCESS]], label %[[EXIT:.+]], label %[[CONT]]
// CHECK: [[EXIT]]
// CHECK: [[NEW_VAL:%.+]] = trunc i64 [[CONV]] to i32
// CHECK: store i32 [[NEW_VAL]], i32* @{{.+}},
#pragma omp atomic capture
  {bfx4.b /= ldv; iv = bfx4.b;}
// CHECK: [[EXPR:%.+]] = load x86_fp80, x86_fp80* @{{.+}}
// CHECK: [[PREV_VALUE:%.+]] = load atomic i8, i8* getelementptr inbounds (%struct.BitFields4_packed, %struct.BitFields4_packed* @{{.+}}, i32 0, i32 0, i64 2) monotonic
// CHECK: br label %[[CONT:.+]]
// CHECK: [[CONT]]
// CHECK: [[OLD_BF_VALUE:%.+]] = phi i8 [ [[PREV_VALUE]], %[[EXIT]] ], [ [[FAILED_OLD_VAL:%.+]], %[[CONT]] ]
// CHECK: [[BITCAST1:%.+]] = bitcast i64* %{{.+}} to i8*
// CHECK: store i8 [[OLD_BF_VALUE]], i8* [[BITCAST1]],
// CHECK: [[BITCAST:%.+]] = bitcast i64* %{{.+}} to i8*
// CHECK: store i8 [[OLD_BF_VALUE]], i8* [[BITCAST]],
// CHECK: [[A_LD:%.+]] = load i8, i8* [[BITCAST]],
// CHECK: [[A_ASHR:%.+]] = ashr i8 [[A_LD]], 1
// CHECK: [[CAST:%.+]] = sext i8 [[A_ASHR]] to i64
// CHECK: [[CONV:%.+]] = sitofp i64 [[CAST]] to x86_fp80
// CHECK: [[ADD:%.+]] = fadd x86_fp80 [[CONV]], [[EXPR]]
// CHECK: [[NEW_VAL:%.+]] = fptosi x86_fp80 [[ADD]] to i64
// CHECK: [[TRUNC:%.+]] = trunc i64 [[NEW_VAL]] to i8
// CHECK: [[BF_LD:%.+]] = load i8, i8* [[BITCAST1]],
// CHECK: [[BF_AND:%.+]] = and i8 [[TRUNC]], 127
// CHECK: [[BF_VALUE:%.+]] = shl i8 [[BF_AND]], 1
// CHECK: [[BF_CLEAR:%.+]] = and i8 [[BF_LD]], 1
// CHECK: or i8 [[BF_CLEAR]], [[BF_VALUE]]
// CHECK: store i8 %{{.+}}, i8* [[BITCAST1]]
// CHECK: [[NEW_BF_VALUE:%.+]] = load i8, i8* [[BITCAST1]]
// CHECK: [[RES:%.+]] = cmpxchg i8* getelementptr inbounds (%struct.BitFields4_packed, %struct.BitFields4_packed* @{{.+}}, i32 0, i32 0, i64 2), i8 [[OLD_BF_VALUE]], i8 [[NEW_BF_VALUE]] monotonic monotonic
// CHECK: [[FAILED_OLD_VAL]] = extractvalue { i8, i1 } [[RES]], 0
// CHECK: [[FAIL_SUCCESS:%.+]] = extractvalue { i8, i1 } [[RES]], 1
// CHECK: br i1 [[FAIL_SUCCESS]], label %[[EXIT:.+]], label %[[CONT]]
// CHECK: [[EXIT]]
// CHECK: [[NEW_VAL_I32:%.+]] = trunc i64 [[NEW_VAL]] to i32
// CHECK: store i32 [[NEW_VAL_I32]], i32* @{{.+}},
#pragma omp atomic capture
  iv = bfx4_packed.b += ldv;
// CHECK: load i64, i64*
// CHECK: [[EXPR:%.+]] = uitofp i64 %{{.+}} to float
// CHECK: [[I64VAL:%.+]] = load atomic i64, i64* bitcast (<2 x float>* [[DEST:@.+]] to i64*) monotonic
// CHECK: br label %[[CONT:.+]]
// CHECK: [[CONT]]
// CHECK: [[OLD_I64:%.+]] = phi i64 [ [[I64VAL]], %{{.+}} ], [ [[FAILED_I64_OLD_VAL:%.+]], %[[CONT]] ]
// CHECK: [[BITCAST:%.+]] = bitcast <2 x float>* [[LDTEMP1:%.+]] to i64*
// CHECK: store i64 [[OLD_I64]], i64* [[BITCAST]],
// CHECK: [[OLD_VEC_VAL:%.+]] = bitcast i64 [[OLD_I64]] to <2 x float>
// CHECK: store <2 x float> [[OLD_VEC_VAL]], <2 x float>* [[LDTEMP:%.+]],
// CHECK: [[VEC_VAL:%.+]] = load <2 x float>, <2 x float>* [[LDTEMP]]
// CHECK: [[X:%.+]] = extractelement <2 x float> [[VEC_VAL]], i64 0
// CHECK: [[VEC_ITEM_VAL:%.+]] = fsub float [[EXPR]], [[X]]
// CHECK: [[VEC_VAL:%.+]] = load <2 x float>, <2 x float>* [[LDTEMP1]],
// CHECK: [[NEW_VEC_VAL:%.+]] = insertelement <2 x float> [[VEC_VAL]], float [[VEC_ITEM_VAL]], i64 0
// CHECK: store <2 x float> [[NEW_VEC_VAL]], <2 x float>* [[LDTEMP1]]
// CHECK: [[NEW_I64:%.+]] = load i64, i64* [[BITCAST]]
// CHECK: [[RES:%.+]] = cmpxchg i64* bitcast (<2 x float>* [[DEST]] to i64*), i64 [[OLD_I64]], i64 [[NEW_I64]] monotonic monotonic
// CHECK: [[FAILED_I64_OLD_VAL:%.+]] = extractvalue { i64, i1 } [[RES]], 0
// CHECK: [[FAIL_SUCCESS:%.+]] = extractvalue { i64, i1 } [[RES]], 1
// CHECK: br i1 [[FAIL_SUCCESS]], label %[[EXIT:.+]], label %[[CONT]]
// CHECK: [[EXIT]]
// CHECK: store float [[X]], float* @{{.+}},
#pragma omp atomic capture
  {fv = float2x.x; float2x.x = ulv - float2x.x;}
// CHECK: [[EXPR:%.+]] = load double, double* @{{.+}},
// CHECK: [[OLD_VAL:%.+]] = call i32 @llvm.read_register.i32([[REG:metadata ![0-9]+]])
// CHECK: [[X_RVAL:%.+]] = sitofp i32 [[OLD_VAL]] to double
// CHECK: [[DIV:%.+]] = fdiv double [[EXPR]], [[X_RVAL]]
// CHECK: [[NEW_VAL:%.+]] = fptosi double [[DIV]] to i32
// CHECK: call void @llvm.write_register.i32([[REG]], i32 [[NEW_VAL]])
// CHECK: store i32 [[NEW_VAL]], i32* @{{.+}},
// CHECK: call{{.*}} @__kmpc_flush(
#pragma omp atomic capture seq_cst
  {rix = dv / rix; iv = rix;}
// CHECK: [[OLD_VAL:%.+]] = atomicrmw xchg i32* @{{.+}}, i32 5 monotonic
// CHECK: call void @llvm.write_register.i32([[REG]], i32 [[OLD_VAL]])
#pragma omp atomic capture
  {rix = ix; ix = 5;}
  return 0;
}
#endif

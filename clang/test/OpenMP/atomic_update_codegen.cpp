// RUN: %clang_cc1 -verify -triple x86_64-apple-darwin10 -target-cpu core2 -fopenmp -x c -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -fopenmp -x c -triple x86_64-apple-darwin10 -target-cpu core2 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -x c -triple x86_64-apple-darwin10 -target-cpu core2 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s

// RUN: %clang_cc1 -verify -triple x86_64-apple-darwin10 -target-cpu core2 -fopenmp-simd -x c -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -fopenmp-simd -x c -triple x86_64-apple-darwin10 -target-cpu core2 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -x c -triple x86_64-apple-darwin10 -target-cpu core2 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// SIMD-ONLY0-NOT: {{__kmpc|__tgt}}
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

int main(void) {
// CHECK: atomicrmw fadd ptr @{{.+}}, double 1.000000e+00 monotonic, align 8
#pragma omp atomic
  ++dv;
// CHECK: atomicrmw add ptr @{{.+}}, i8 1 monotonic, align 1
#pragma omp atomic
  bx++;
// CHECK: atomicrmw add ptr @{{.+}}, i8 1 monotonic, align 1
#pragma omp atomic update
  ++cx;
// CHECK: atomicrmw sub ptr @{{.+}}, i8 1 monotonic, align 1
#pragma omp atomic
  ucx--;
// CHECK: atomicrmw sub ptr @{{.+}}, i16 1 monotonic, align 2
#pragma omp atomic update
  --sx;
// CHECK: [[USV:%.+]] = load i16, ptr @{{.+}},
// CHECK: [[EXPR:%.+]] = zext i16 [[USV]] to i32
// CHECK: [[X:%.+]] = load atomic i16, ptr [[X_ADDR:@.+]] monotonic, align 2
// CHECK: br label %[[CONT:.+]]
// CHECK: [[CONT]]
// CHECK: [[EXPECTED:%.+]] = phi i16 [ [[X]], %{{.+}} ], [ [[OLD_X:%.+]], %[[CONT]] ]
// CHECK: [[CONV:%.+]] = zext i16 [[EXPECTED]] to i32
// CHECK: [[ADD:%.+]] = add nsw i32 [[CONV]], [[EXPR]]
// CHECK: [[DESIRED:%.+]] = trunc i32 [[ADD]] to i16
// CHECK: store i16 [[DESIRED]], ptr [[TEMP:%.+]],
// CHECK: [[DESIRED:%.+]] = load i16, ptr [[TEMP]],
// CHECK: [[RES:%.+]] = cmpxchg ptr [[X_ADDR]], i16 [[EXPECTED]], i16 [[DESIRED]] monotonic monotonic, align 2
// CHECK: [[OLD_X]] = extractvalue { i16, i1 } [[RES]], 0
// CHECK: [[SUCCESS_FAIL:%.+]] = extractvalue { i16, i1 } [[RES]], 1
// CHECK: br i1 [[SUCCESS_FAIL]], label %[[EXIT:.+]], label %[[CONT]]
// CHECK: [[EXIT]]
#pragma omp atomic
  usx += usv;
// CHECK: [[EXPR:%.+]] = load i32, ptr @{{.+}},
// CHECK: [[X:%.+]] = load atomic i32, ptr [[X_ADDR:@.+]] monotonic, align 4
// CHECK: br label %[[CONT:.+]]
// CHECK: [[CONT]]
// CHECK: [[EXPECTED:%.+]] = phi i32 [ [[X]], %{{.+}} ], [ [[OLD_X:%.+]], %[[CONT]] ]
// CHECK: [[DESIRED:%.+]] = mul nsw i32 [[EXPECTED]], [[EXPR]]
// CHECK: store i32 [[DESIRED]], ptr [[TEMP:%.+]],
// CHECK: [[DESIRED:%.+]] = load i32, ptr [[TEMP]],
// CHECK: [[RES:%.+]] = cmpxchg ptr [[X_ADDR]], i32 [[EXPECTED]], i32 [[DESIRED]] monotonic monotonic, align 4
// CHECK: [[OLD_X]] = extractvalue { i32, i1 } [[RES]], 0
// CHECK: [[SUCCESS_FAIL:%.+]] = extractvalue { i32, i1 } [[RES]], 1
// CHECK: br i1 [[SUCCESS_FAIL]], label %[[EXIT:.+]], label %[[CONT]]
// CHECK: [[EXIT]]
#pragma omp atomic update
  ix *= iv;
// CHECK: [[EXPR:%.+]] = load i32, ptr @{{.+}},
// CHECK: atomicrmw sub ptr @{{.+}}, i32 [[EXPR]] monotonic, align 4
#pragma omp atomic
  uix -= uiv;
// CHECK: [[EXPR:%.+]] = load i32, ptr @{{.+}},
// CHECK: [[X:%.+]] = load atomic i32, ptr [[X_ADDR:@.+]] monotonic, align 4
// CHECK: br label %[[CONT:.+]]
// CHECK: [[CONT]]
// CHECK: [[EXPECTED:%.+]] = phi i32 [ [[X]], %{{.+}} ], [ [[OLD_X:%.+]], %[[CONT]] ]
// CHECK: [[DESIRED:%.+]] = shl i32 [[EXPECTED]], [[EXPR]]
// CHECK: store i32 [[DESIRED]], ptr [[TEMP:%.+]],
// CHECK: [[DESIRED:%.+]] = load i32, ptr [[TEMP]],
// CHECK: [[RES:%.+]] = cmpxchg ptr [[X_ADDR]], i32 [[EXPECTED]], i32 [[DESIRED]] monotonic monotonic, align 4
// CHECK: [[OLD_X]] = extractvalue { i32, i1 } [[RES]], 0
// CHECK: [[SUCCESS_FAIL:%.+]] = extractvalue { i32, i1 } [[RES]], 1
// CHECK: br i1 [[SUCCESS_FAIL]], label %[[EXIT:.+]], label %[[CONT]]
// CHECK: [[EXIT]]
#pragma omp atomic update
  ix <<= iv;
// CHECK: [[EXPR:%.+]] = load i32, ptr @{{.+}},
// CHECK: [[X:%.+]] = load atomic i32, ptr [[X_ADDR:@.+]] monotonic, align 4
// CHECK: br label %[[CONT:.+]]
// CHECK: [[CONT]]
// CHECK: [[EXPECTED:%.+]] = phi i32 [ [[X]], %{{.+}} ], [ [[OLD_X:%.+]], %[[CONT]] ]
// CHECK: [[DESIRED:%.+]] = lshr i32 [[EXPECTED]], [[EXPR]]
// CHECK: store i32 [[DESIRED]], ptr [[TEMP:%.+]],
// CHECK: [[DESIRED:%.+]] = load i32, ptr [[TEMP]],
// CHECK: [[RES:%.+]] = cmpxchg ptr [[X_ADDR]], i32 [[EXPECTED]], i32 [[DESIRED]] monotonic monotonic, align 4
// CHECK: [[OLD_X]] = extractvalue { i32, i1 } [[RES]], 0
// CHECK: [[SUCCESS_FAIL:%.+]] = extractvalue { i32, i1 } [[RES]], 1
// CHECK: br i1 [[SUCCESS_FAIL]], label %[[EXIT:.+]], label %[[CONT]]
// CHECK: [[EXIT]]
#pragma omp atomic
  uix >>= uiv;
// CHECK: [[EXPR:%.+]] = load i64, ptr @{{.+}},
// CHECK: [[X:%.+]] = load atomic i64, ptr [[X_ADDR:@.+]] monotonic, align 8
// CHECK: br label %[[CONT:.+]]
// CHECK: [[CONT]]
// CHECK: [[EXPECTED:%.+]] = phi i64 [ [[X]], %{{.+}} ], [ [[OLD_X:%.+]], %[[CONT]] ]
// CHECK: [[DESIRED:%.+]] = sdiv i64 [[EXPECTED]], [[EXPR]]
// CHECK: store i64 [[DESIRED]], ptr [[TEMP:%.+]],
// CHECK: [[DESIRED:%.+]] = load i64, ptr [[TEMP]],
// CHECK: [[RES:%.+]] = cmpxchg ptr [[X_ADDR]], i64 [[EXPECTED]], i64 [[DESIRED]] monotonic monotonic, align 8
// CHECK: [[OLD_X]] = extractvalue { i64, i1 } [[RES]], 0
// CHECK: [[SUCCESS_FAIL:%.+]] = extractvalue { i64, i1 } [[RES]], 1
// CHECK: br i1 [[SUCCESS_FAIL]], label %[[EXIT:.+]], label %[[CONT]]
// CHECK: [[EXIT]]
#pragma omp atomic update
  lx /= lv;
// CHECK: [[EXPR:%.+]] = load i64, ptr @{{.+}},
// CHECK: atomicrmw and ptr @{{.+}}, i64 [[EXPR]] monotonic, align 8
#pragma omp atomic
  ulx &= ulv;
// CHECK: [[EXPR:%.+]] = load i64, ptr @{{.+}},
// CHECK: atomicrmw xor ptr @{{.+}}, i64 [[EXPR]] monotonic, align 8
#pragma omp atomic update
  llx ^= llv;
// CHECK: [[EXPR:%.+]] = load i64, ptr @{{.+}},
// CHECK: atomicrmw or ptr @{{.+}}, i64 [[EXPR]] monotonic, align 8
#pragma omp atomic
  ullx |= ullv;
// CHECK: [[EXPR:%.+]] = load float, ptr @{{.+}},
// CHECK: atomicrmw fadd ptr @{{.+}}, float [[EXPR]] monotonic, align 4
#pragma omp atomic update
  fx = fx + fv;
// CHECK: [[EXPR:%.+]] = load double, ptr @{{.+}},
// CHECK: [[OLD:%.+]] = load atomic i64, ptr [[X_ADDR:@.+]] monotonic, align 8
// CHECK: br label %[[CONT:.+]]
// CHECK: [[CONT]]
// CHECK: [[EXPECTED:%.+]] = phi i64 [ [[OLD]], %{{.+}} ], [ [[PREV:%.+]], %[[CONT]] ]
// CHECK: [[OLD:%.+]] = bitcast i64 [[EXPECTED]] to double
// CHECK: [[SUB:%.+]] = fsub double [[EXPR]], [[OLD]]
// CHECK: store double [[SUB]], ptr [[TEMP:%.+]],
// CHECK: [[DESIRED:%.+]] = load i64, ptr [[TEMP]],
// CHECK: [[RES:%.+]] = cmpxchg ptr [[X_ADDR]], i64 [[EXPECTED]], i64 [[DESIRED]] monotonic monotonic, align 8
// CHECK: [[PREV:%.+]] = extractvalue { i64, i1 } [[RES]], 0
// CHECK: [[SUCCESS_FAIL:%.+]] = extractvalue { i64, i1 } [[RES]], 1
// CHECK: br i1 [[SUCCESS_FAIL]], label %[[EXIT:.+]], label %[[CONT]]
// CHECK: [[EXIT]]
#pragma omp atomic
  dx = dv - dx;
// CHECK: [[EXPR:%.+]] = load x86_fp80, ptr @{{.+}},
// CHECK: [[OLD:%.+]] = load atomic i128, ptr [[X_ADDR:@.+]] monotonic, align 16
// CHECK: br label %[[CONT:.+]]
// CHECK: [[CONT]]
// CHECK: [[EXPECTED:%.+]] = phi i128 [ [[OLD]], %{{.+}} ], [ [[PREV:%.+]], %[[CONT]] ]
// CHECK: store i128 [[EXPECTED]], ptr [[TEMP:%.+]],
// CHECK: store i128 [[EXPECTED]], ptr [[TEMP1:%.+]],
// CHECK: [[OLD:%.+]] = load x86_fp80, ptr [[TEMP1]]
// CHECK: [[MUL:%.+]] = fmul x86_fp80 [[OLD]], [[EXPR]]
// CHECK: store x86_fp80 [[MUL]], ptr [[TEMP]]
// CHECK: [[DESIRED:%.+]] = load i128, ptr [[TEMP]]
// CHECK: [[RES:%.+]] = cmpxchg ptr [[X_ADDR]], i128 [[EXPECTED]], i128 [[DESIRED]] monotonic monotonic, align 16
// CHECK: [[PREV:%.+]] = extractvalue { i128, i1 } [[RES]], 0
// CHECK: [[SUCCESS_FAIL:%.+]] = extractvalue { i128, i1 } [[RES]], 1
// CHECK: br i1 [[SUCCESS_FAIL]], label %[[EXIT:.+]], label %[[CONT]]
// CHECK: [[EXIT]]
#pragma omp atomic update
  ldx = ldx * ldv;
// CHECK: [[EXPR_RE:%.+]] = load i32, ptr @{{.+}}
// CHECK: [[EXPR_IM:%.+]] = load i32, ptr getelementptr inbounds ({ i32, i32 }, ptr @{{.+}}, i32 0, i32 1)
// CHECK: call void @__atomic_load(i64 noundef 8, ptr noundef [[X_ADDR:@.+]], ptr noundef [[EXPECTED_ADDR:%.+]], i32 noundef 0)
// CHECK: br label %[[CONT:.+]]
// CHECK: [[CONT]]
// CHECK: [[X_RE_ADDR:%.+]] = getelementptr inbounds { i32, i32 }, ptr [[EXPECTED_ADDR]], i32 0, i32 0
// CHECK: [[X_RE:%.+]] = load i32, ptr [[X_RE_ADDR]]
// CHECK: [[X_IM_ADDR:%.+]] = getelementptr inbounds { i32, i32 }, ptr [[EXPECTED_ADDR]], i32 0, i32 1
// CHECK: [[X_IM:%.+]] = load i32, ptr [[X_IM_ADDR]]
// <Skip checks for complex calculations>
// CHECK: [[X_RE_ADDR:%.+]] = getelementptr inbounds { i32, i32 }, ptr [[DESIRED_ADDR:%.+]], i32 0, i32 0
// CHECK: [[X_IM_ADDR:%.+]] = getelementptr inbounds { i32, i32 }, ptr [[DESIRED_ADDR]], i32 0, i32 1
// CHECK: store i32 %{{.+}}, ptr [[X_RE_ADDR]]
// CHECK: store i32 %{{.+}}, ptr [[X_IM_ADDR]]
// CHECK: [[SUCCESS_FAIL:%.+]] = call zeroext i1 @__atomic_compare_exchange(i64 noundef 8, ptr noundef [[X_ADDR]], ptr noundef [[EXPECTED_ADDR]], ptr noundef [[DESIRED_ADDR]], i32 noundef 0, i32 noundef 0)
// CHECK: br i1 [[SUCCESS_FAIL]], label %[[EXIT:.+]], label %[[CONT]]
// CHECK: [[EXIT]]
#pragma omp atomic
  cix = civ / cix;
// CHECK: [[EXPR_RE:%.+]] = load float, ptr @{{.+}}
// CHECK: [[EXPR_IM:%.+]] = load float, ptr getelementptr inbounds ({ float, float }, ptr @{{.+}}, i32 0, i32 1)
// CHECK: call void @__atomic_load(i64 noundef 8, ptr noundef [[X_ADDR:@.+]], ptr noundef [[EXPECTED_ADDR:%.+]], i32 noundef 0)
// CHECK: br label %[[CONT:.+]]
// CHECK: [[CONT]]
// CHECK: [[X_RE_ADDR:%.+]] = getelementptr inbounds { float, float }, ptr [[EXPECTED_ADDR]], i32 0, i32 0
// CHECK: [[X_RE:%.+]] = load float, ptr [[X_RE_ADDR]]
// CHECK: [[X_IM_ADDR:%.+]] = getelementptr inbounds { float, float }, ptr [[EXPECTED_ADDR]], i32 0, i32 1
// CHECK: [[X_IM:%.+]] = load float, ptr [[X_IM_ADDR]]
// <Skip checks for complex calculations>
// CHECK: [[X_RE_ADDR:%.+]] = getelementptr inbounds { float, float }, ptr [[DESIRED_ADDR:%.+]], i32 0, i32 0
// CHECK: [[X_IM_ADDR:%.+]] = getelementptr inbounds { float, float }, ptr [[DESIRED_ADDR]], i32 0, i32 1
// CHECK: store float %{{.+}}, ptr [[X_RE_ADDR]]
// CHECK: store float %{{.+}}, ptr [[X_IM_ADDR]]
// CHECK: [[SUCCESS_FAIL:%.+]] = call zeroext i1 @__atomic_compare_exchange(i64 noundef 8, ptr noundef [[X_ADDR]], ptr noundef [[EXPECTED_ADDR]], ptr noundef [[DESIRED_ADDR]], i32 noundef 0, i32 noundef 0)
// CHECK: br i1 [[SUCCESS_FAIL]], label %[[EXIT:.+]], label %[[CONT]]
// CHECK: [[EXIT]]
#pragma omp atomic update
  cfx = cfv + cfx;
// CHECK: [[EXPR_RE:%.+]] = load double, ptr @{{.+}}
// CHECK: [[EXPR_IM:%.+]] = load double, ptr getelementptr inbounds ({ double, double }, ptr @{{.+}}, i32 0, i32 1)
// CHECK: call void @__atomic_load(i64 noundef 16, ptr noundef [[X_ADDR:@.+]], ptr noundef [[EXPECTED_ADDR:%.+]], i32 noundef 5)
// CHECK: br label %[[CONT:.+]]
// CHECK: [[CONT]]
// CHECK: [[X_RE_ADDR:%.+]] = getelementptr inbounds { double, double }, ptr [[EXPECTED_ADDR]], i32 0, i32 0
// CHECK: [[X_RE:%.+]] = load double, ptr [[X_RE_ADDR]]
// CHECK: [[X_IM_ADDR:%.+]] = getelementptr inbounds { double, double }, ptr [[EXPECTED_ADDR]], i32 0, i32 1
// CHECK: [[X_IM:%.+]] = load double, ptr [[X_IM_ADDR]]
// <Skip checks for complex calculations>
// CHECK: [[X_RE_ADDR:%.+]] = getelementptr inbounds { double, double }, ptr [[DESIRED_ADDR:%.+]], i32 0, i32 0
// CHECK: [[X_IM_ADDR:%.+]] = getelementptr inbounds { double, double }, ptr [[DESIRED_ADDR]], i32 0, i32 1
// CHECK: store double %{{.+}}, ptr [[X_RE_ADDR]]
// CHECK: store double %{{.+}}, ptr [[X_IM_ADDR]]
// CHECK: [[SUCCESS_FAIL:%.+]] = call zeroext i1 @__atomic_compare_exchange(i64 noundef 16, ptr noundef [[X_ADDR]], ptr noundef [[EXPECTED_ADDR]], ptr noundef [[DESIRED_ADDR]], i32 noundef 5, i32 noundef 5)
// CHECK: br i1 [[SUCCESS_FAIL]], label %[[EXIT:.+]], label %[[CONT]]
// CHECK: [[EXIT]]
// CHECK: call{{.*}} @__kmpc_flush(
#pragma omp atomic seq_cst
  cdx = cdx - cdv;
// CHECK: [[BV:%.+]] = load i8, ptr @{{.+}}
// CHECK: [[BOOL:%.+]] = trunc i8 [[BV]] to i1
// CHECK: [[EXPR:%.+]] = zext i1 [[BOOL]] to i64
// CHECK: atomicrmw and ptr @{{.+}}, i64 [[EXPR]] monotonic, align 8
#pragma omp atomic update
  ulx = ulx & bv;
// CHECK: [[CV:%.+]]  = load i8, ptr @{{.+}}, align 1
// CHECK: [[EXPR:%.+]] = sext i8 [[CV]] to i32
// CHECK: [[BX:%.+]] = load atomic i8, ptr [[BX_ADDR:@.+]] monotonic, align 1
// CHECK: br label %[[CONT:.+]]
// CHECK: [[CONT]]
// CHECK: [[EXPECTED:%.+]] = phi i8 [ [[BX]], %{{.+}} ], [ [[OLD_X:%.+]], %[[CONT]] ]
// CHECK: [[OLD:%.+]] = trunc i8 [[EXPECTED]] to i1
// CHECK: [[X_RVAL:%.+]] = zext i1 [[OLD]] to i32
// CHECK: [[AND:%.+]] = and i32 [[EXPR]], [[X_RVAL]]
// CHECK: [[CAST:%.+]] = icmp ne i32 [[AND]], 0
// CHECK: [[DESIRED:%.+]] = zext i1 [[CAST]] to i8
// CHECK: store i8 [[DESIRED]], ptr [[TEMP:%.+]],
// CHECK: [[DESIRED:%.+]] = load i8, ptr [[TEMP]],
// CHECK: [[RES:%.+]] = cmpxchg ptr [[BX_ADDR]], i8 [[EXPECTED]], i8 [[DESIRED]] monotonic monotonic, align 1
// CHECK: [[OLD_X:%.+]] = extractvalue { i8, i1 } [[RES]], 0
// CHECK: [[SUCCESS_FAIL:%.+]] = extractvalue { i8, i1 } [[RES]], 1
// CHECK: br i1 [[SUCCESS_FAIL]], label %[[EXIT:.+]], label %[[CONT]]
// CHECK: [[EXIT]]
#pragma omp atomic
  bx = cv & bx;
// CHECK: [[UCV:%.+]]  = load i8, ptr @{{.+}},
// CHECK: [[EXPR:%.+]] = zext i8 [[UCV]] to i32
// CHECK: [[X:%.+]] = load atomic i8, ptr [[CX_ADDR:@.+]] seq_cst, align 1
// CHECK: br label %[[CONT:.+]]
// CHECK: [[CONT]]
// CHECK: [[EXPECTED:%.+]] = phi i8 [ [[X]], %{{.+}} ], [ [[OLD_X:%.+]], %[[CONT]] ]
// CHECK: [[X_RVAL:%.+]] = sext i8 [[EXPECTED]] to i32
// CHECK: [[ASHR:%.+]] = ashr i32 [[X_RVAL]], [[EXPR]]
// CHECK: [[DESIRED:%.+]] = trunc i32 [[ASHR]] to i8
// CHECK: store i8 [[DESIRED]], ptr [[TEMP:%.+]],
// CHECK: [[DESIRED:%.+]] = load i8, ptr [[TEMP]],
// CHECK: [[RES:%.+]] = cmpxchg ptr [[CX_ADDR]], i8 [[EXPECTED]], i8 [[DESIRED]] seq_cst seq_cst, align 1
// CHECK: [[OLD_X:%.+]] = extractvalue { i8, i1 } [[RES]], 0
// CHECK: [[SUCCESS_FAIL:%.+]] = extractvalue { i8, i1 } [[RES]], 1
// CHECK: br i1 [[SUCCESS_FAIL]], label %[[EXIT:.+]], label %[[CONT]]
// CHECK: [[EXIT]]
// CHECK: call{{.*}} @__kmpc_flush(
#pragma omp atomic update, seq_cst
  cx = cx >> ucv;
// CHECK: [[SV:%.+]]  = load i16, ptr @{{.+}},
// CHECK: [[EXPR:%.+]] = sext i16 [[SV]] to i32
// CHECK: [[X:%.+]] = load atomic i64, ptr [[ULX_ADDR:@.+]] monotonic, align 8
// CHECK: br label %[[CONT:.+]]
// CHECK: [[CONT]]
// CHECK: [[EXPECTED:%.+]] = phi i64 [ [[X]], %{{.+}} ], [ [[OLD_X:%.+]], %[[CONT]] ]
// CHECK: [[X_RVAL:%.+]] = trunc i64 [[EXPECTED]] to i32
// CHECK: [[SHL:%.+]] = shl i32 [[EXPR]], [[X_RVAL]]
// CHECK: [[DESIRED:%.+]] = sext i32 [[SHL]] to i64
// CHECK: store i64 [[DESIRED]], ptr [[TEMP:%.+]],
// CHECK: [[DESIRED:%.+]] = load i64, ptr [[TEMP]],
// CHECK: [[RES:%.+]] = cmpxchg ptr [[ULX_ADDR]], i64 [[EXPECTED]], i64 [[DESIRED]] monotonic monotonic, align 8
// CHECK: [[OLD_X:%.+]] = extractvalue { i64, i1 } [[RES]], 0
// CHECK: [[SUCCESS_FAIL:%.+]] = extractvalue { i64, i1 } [[RES]], 1
// CHECK: br i1 [[SUCCESS_FAIL]], label %[[EXIT:.+]], label %[[CONT]]
// CHECK: [[EXIT]]
#pragma omp atomic update
  ulx = sv << ulx;
// CHECK: [[USV:%.+]]  = load i16, ptr @{{.+}},
// CHECK: [[EXPR:%.+]] = zext i16 [[USV]] to i64
// CHECK: [[X:%.+]] = load atomic i64, ptr [[LX_ADDR:@.+]] monotonic, align 8
// CHECK: br label %[[CONT:.+]]
// CHECK: [[CONT]]
// CHECK: [[EXPECTED:%.+]] = phi i64 [ [[X]], %{{.+}} ], [ [[OLD_X:%.+]], %[[CONT]] ]
// CHECK: [[DESIRED:%.+]] = srem i64 [[EXPECTED]], [[EXPR]]
// CHECK: store i64 [[DESIRED]], ptr [[TEMP:%.+]],
// CHECK: [[DESIRED:%.+]] = load i64, ptr [[TEMP]],
// CHECK: [[RES:%.+]] = cmpxchg ptr [[LX_ADDR]], i64 [[EXPECTED]], i64 [[DESIRED]] monotonic monotonic, align 8
// CHECK: [[OLD_X:%.+]] = extractvalue { i64, i1 } [[RES]], 0
// CHECK: [[SUCCESS_FAIL:%.+]] = extractvalue { i64, i1 } [[RES]], 1
// CHECK: br i1 [[SUCCESS_FAIL]], label %[[EXIT:.+]], label %[[CONT]]
// CHECK: [[EXIT]]
#pragma omp atomic
  lx = lx % usv;
// CHECK: [[EXPR:%.+]] = load i32, ptr @{{.+}}
// CHECK: atomicrmw or ptr @{{.+}}, i32 [[EXPR]] seq_cst, align 4
// CHECK: call{{.*}} @__kmpc_flush(
#pragma omp atomic seq_cst, update
  uix = iv | uix;
// CHECK: [[EXPR:%.+]] = load i32, ptr @{{.+}}
// CHECK: atomicrmw and ptr @{{.+}}, i32 [[EXPR]] monotonic, align 4
#pragma omp atomic
  ix = ix & uiv;
// CHECK: [[EXPR:%.+]] = load i64, ptr @{{.+}},
// CHECK: call void @__atomic_load(i64 noundef 8, ptr noundef [[X_ADDR:@.+]], ptr noundef [[EXPECTED_ADDR:%.+]], i32 noundef 0)
// CHECK: br label %[[CONT:.+]]
// CHECK: [[CONT]]
// CHECK: [[X_RE_ADDR:%.+]] = getelementptr inbounds { i32, i32 }, ptr [[EXPECTED_ADDR]], i32 0, i32 0
// CHECK: [[X_RE:%.+]] = load i32, ptr [[X_RE_ADDR]]
// CHECK: [[X_IM_ADDR:%.+]] = getelementptr inbounds { i32, i32 }, ptr [[EXPECTED_ADDR]], i32 0, i32 1
// CHECK: [[X_IM:%.+]] = load i32, ptr [[X_IM_ADDR]]
// <Skip checks for complex calculations>
// CHECK: [[X_RE_ADDR:%.+]] = getelementptr inbounds { i32, i32 }, ptr [[DESIRED_ADDR:%.+]], i32 0, i32 0
// CHECK: [[X_IM_ADDR:%.+]] = getelementptr inbounds { i32, i32 }, ptr [[DESIRED_ADDR]], i32 0, i32 1
// CHECK: store i32 %{{.+}}, ptr [[X_RE_ADDR]]
// CHECK: store i32 %{{.+}}, ptr [[X_IM_ADDR]]
// CHECK: [[SUCCESS_FAIL:%.+]] = call zeroext i1 @__atomic_compare_exchange(i64 noundef 8, ptr noundef [[X_ADDR]], ptr noundef [[EXPECTED_ADDR]], ptr noundef [[DESIRED_ADDR]], i32 noundef 0, i32 noundef 0)
// CHECK: br i1 [[SUCCESS_FAIL]], label %[[EXIT:.+]], label %[[CONT]]
// CHECK: [[EXIT]]
#pragma omp atomic update
  cix = lv + cix;
// CHECK: [[ULV:%.+]] = load i64, ptr @{{.+}},
// CHECK: [[EXPR:%.+]] = uitofp i64 [[ULV]] to float
// CHECK: [[OLD:%.+]] = load atomic i32, ptr [[X_ADDR:@.+]] monotonic, align 4
// CHECK: br label %[[CONT:.+]]
// CHECK: [[CONT]]
// CHECK: [[EXPECTED:%.+]] = phi i32 [ [[OLD]], %{{.+}} ], [ [[PREV:%.+]], %[[CONT]] ]
// CHECK: [[OLD:%.+]] = bitcast i32 [[EXPECTED]] to float
// CHECK: [[MUL:%.+]] = fmul float [[OLD]], [[EXPR]]
// CHECK: store float [[MUL]], ptr [[TEMP:%.+]],
// CHECK: [[DESIRED:%.+]] = load i32, ptr [[TEMP]],
// CHECK: [[RES:%.+]] = cmpxchg ptr [[X_ADDR]], i32 [[EXPECTED]], i32 [[DESIRED]] monotonic monotonic, align 4
// CHECK: [[PREV:%.+]] = extractvalue { i32, i1 } [[RES]], 0
// CHECK: [[SUCCESS_FAIL:%.+]] = extractvalue { i32, i1 } [[RES]], 1
// CHECK: br i1 [[SUCCESS_FAIL]], label %[[EXIT:.+]], label %[[CONT]]
// CHECK: [[EXIT]]
#pragma omp atomic
  fx = fx * ulv;
// CHECK: [[LLV:%.+]] = load i64, ptr @{{.+}},
// CHECK: [[EXPR:%.+]] = sitofp i64 [[LLV]] to double
// CHECK: [[OLD:%.+]] = load atomic i64, ptr [[X_ADDR:@.+]] monotonic, align 8
// CHECK: br label %[[CONT:.+]]
// CHECK: [[CONT]]
// CHECK: [[EXPECTED:%.+]] = phi i64 [ [[OLD]], %{{.+}} ], [ [[PREV:%.+]], %[[CONT]] ]
// CHECK: [[OLD:%.+]] = bitcast i64 [[EXPECTED]] to double
// CHECK: [[DIV:%.+]] = fdiv double [[OLD]], [[EXPR]]
// CHECK: store double [[DIV]], ptr [[TEMP:%.+]],
// CHECK: [[DESIRED:%.+]] = load i64, ptr [[TEMP]],
// CHECK: [[RES:%.+]] = cmpxchg ptr [[X_ADDR]], i64 [[EXPECTED]], i64 [[DESIRED]] monotonic monotonic, align 8
// CHECK: [[PREV:%.+]] = extractvalue { i64, i1 } [[RES]], 0
// CHECK: [[SUCCESS_FAIL:%.+]] = extractvalue { i64, i1 } [[RES]], 1
// CHECK: br i1 [[SUCCESS_FAIL]], label %[[EXIT:.+]], label %[[CONT]]
// CHECK: [[EXIT]]
#pragma omp atomic update
  dx /= llv;
// CHECK: [[ULLV:%.+]] = load i64, ptr @{{.+}},
// CHECK: [[EXPR:%.+]] = uitofp i64 [[ULLV]] to x86_fp80
// CHECK: [[OLD:%.+]] = load atomic i128, ptr [[X_ADDR:@.+]] monotonic, align 16
// CHECK: br label %[[CONT:.+]]
// CHECK: [[CONT]]
// CHECK: [[EXPECTED:%.+]] = phi i128 [ [[OLD]], %{{.+}} ], [ [[PREV:%.+]], %[[CONT]] ]
// CHECK: store i128 [[EXPECTED]], ptr [[TEMP1:%.+]]
// CHECK: [[OLD:%.+]] = load x86_fp80, ptr [[TEMP:%.+]]
// CHECK: [[SUB:%.+]] = fsub x86_fp80 [[OLD]], [[EXPR]]
// CHECK: store x86_fp80 [[SUB]], ptr [[TEMP1]]
// CHECK: [[DESIRED:%.+]] = load i128, ptr [[TEMP1]]
// CHECK: [[RES:%.+]] = cmpxchg ptr [[X_ADDR]], i128 [[EXPECTED]], i128 [[DESIRED]] monotonic monotonic, align 16
// CHECK: [[PREV:%.+]] = extractvalue { i128, i1 } [[RES]], 0
// CHECK: [[SUCCESS_FAIL:%.+]] = extractvalue { i128, i1 } [[RES]], 1
// CHECK: br i1 [[SUCCESS_FAIL]], label %[[EXIT:.+]], label %[[CONT]]
// CHECK: [[EXIT]]
#pragma omp atomic
  ldx -= ullv;
// CHECK: [[EXPR:%.+]] = load float, ptr @{{.+}},
// CHECK: call void @__atomic_load(i64 noundef 8, ptr noundef [[X_ADDR:@.+]], ptr noundef [[EXPECTED_ADDR:%.+]], i32 noundef 0)
// CHECK: br label %[[CONT:.+]]
// CHECK: [[CONT]]
// CHECK: [[X_RE_ADDR:%.+]] = getelementptr inbounds { i32, i32 }, ptr [[EXPECTED_ADDR]], i32 0, i32 0
// CHECK: [[X_RE:%.+]] = load i32, ptr [[X_RE_ADDR]]
// CHECK: [[X_IM_ADDR:%.+]] = getelementptr inbounds { i32, i32 }, ptr [[EXPECTED_ADDR]], i32 0, i32 1
// CHECK: [[X_IM:%.+]] = load i32, ptr [[X_IM_ADDR]]
// <Skip checks for complex calculations>
// CHECK: [[X_RE_ADDR:%.+]] = getelementptr inbounds { i32, i32 }, ptr [[DESIRED_ADDR:%.+]], i32 0, i32 0
// CHECK: [[X_IM_ADDR:%.+]] = getelementptr inbounds { i32, i32 }, ptr [[DESIRED_ADDR]], i32 0, i32 1
// CHECK: store i32 %{{.+}}, ptr [[X_RE_ADDR]]
// CHECK: store i32 %{{.+}}, ptr [[X_IM_ADDR]]
// CHECK: [[SUCCESS_FAIL:%.+]] = call zeroext i1 @__atomic_compare_exchange(i64 noundef 8, ptr noundef [[X_ADDR]], ptr noundef [[EXPECTED_ADDR]], ptr noundef [[DESIRED_ADDR]], i32 noundef 0, i32 noundef 0)
// CHECK: br i1 [[SUCCESS_FAIL]], label %[[EXIT:.+]], label %[[CONT]]
// CHECK: [[EXIT]]
#pragma omp atomic update
  cix = fv / cix;
// CHECK: [[EXPR:%.+]] = load double, ptr @{{.+}},
// CHECK: [[X:%.+]] = load atomic i16, ptr [[X_ADDR:@.+]] monotonic, align 2
// CHECK: br label %[[CONT:.+]]
// CHECK: [[CONT]]
// CHECK: [[EXPECTED:%.+]] = phi i16 [ [[X]], %{{.+}} ], [ [[OLD_X:%.+]], %[[CONT]] ]
// CHECK: [[CONV:%.+]] = sext i16 [[EXPECTED]] to i32
// CHECK: [[X_RVAL:%.+]] = sitofp i32 [[CONV]] to double
// CHECK: [[ADD:%.+]] = fadd double [[X_RVAL]], [[EXPR]]
// CHECK: [[DESIRED:%.+]] = fptosi double [[ADD]] to i16
// CHECK: store i16 [[DESIRED]], ptr [[TEMP:%.+]]
// CHECK: [[DESIRED:%.+]] = load i16, ptr [[TEMP]]
// CHECK: [[RES:%.+]] = cmpxchg ptr [[X_ADDR]], i16 [[EXPECTED]], i16 [[DESIRED]] monotonic monotonic, align 2
// CHECK: [[OLD_X]] = extractvalue { i16, i1 } [[RES]], 0
// CHECK: [[SUCCESS_FAIL:%.+]] = extractvalue { i16, i1 } [[RES]], 1
// CHECK: br i1 [[SUCCESS_FAIL]], label %[[EXIT:.+]], label %[[CONT]]
// CHECK: [[EXIT]]
#pragma omp atomic
  sx = sx + dv;
// CHECK: [[EXPR:%.+]] = load x86_fp80, ptr @{{.+}},
// CHECK: [[XI8:%.+]] = load atomic i8, ptr [[X_ADDR:@.+]] monotonic, align 1
// CHECK: br label %[[CONT:.+]]
// CHECK: [[CONT]]
// CHECK: [[EXPECTED:%.+]] = phi i8 [ [[XI8]], %{{.+}} ], [ [[OLD_XI8:%.+]], %[[CONT]] ]
// CHECK: [[BOOL_EXPECTED:%.+]] = trunc i8 [[EXPECTED]] to i1
// CHECK: [[CONV:%.+]] = zext i1 [[BOOL_EXPECTED]] to i32
// CHECK: [[X_RVAL:%.+]] = sitofp i32 [[CONV]] to x86_fp80
// CHECK: [[MUL:%.+]] = fmul x86_fp80 [[EXPR]], [[X_RVAL]]
// CHECK: [[BOOL_DESIRED:%.+]] = fcmp une x86_fp80 [[MUL]], 0xK00000000000000000000
// CHECK: [[DESIRED:%.+]] = zext i1 [[BOOL_DESIRED]] to i8
// CHECK: store i8 [[DESIRED]], ptr [[TEMP:%.+]]
// CHECK: [[DESIRED:%.+]] = load i8, ptr [[TEMP]]
// CHECK: [[RES:%.+]] = cmpxchg ptr [[X_ADDR]], i8 [[EXPECTED]], i8 [[DESIRED]] release monotonic, align 1
// CHECK: [[OLD_XI8:%.+]] = extractvalue { i8, i1 } [[RES]], 0
// CHECK: [[SUCCESS_FAIL:%.+]] = extractvalue { i8, i1 } [[RES]], 1
// CHECK: br i1 [[SUCCESS_FAIL]], label %[[EXIT:.+]], label %[[CONT]]
// CHECK: [[EXIT]]
// CHECK: call{{.*}} @__kmpc_flush(
#pragma omp atomic update release
  bx = ldv * bx;
// CHECK: [[EXPR_RE:%.+]] = load i32, ptr [[CIV_ADDR:@.+]],
// CHECK: [[EXPR_IM:%.+]] = load i32, ptr getelementptr inbounds ({ i32, i32 }, ptr [[CIV_ADDR]], i32 0, i32 1),
// CHECK: [[XI8:%.+]] = load atomic i8, ptr [[X_ADDR:@.+]] monotonic, align 1
// CHECK: br label %[[CONT:.+]]
// CHECK: [[CONT]]
// CHECK: [[EXPECTED:%.+]] = phi i8 [ [[XI8]], %{{.+}} ], [ [[OLD_XI8:%.+]], %[[CONT]] ]
// CHECK: [[BOOL_EXPECTED:%.+]] = trunc i8 [[EXPECTED]] to i1
// CHECK: [[X_RVAL:%.+]] = zext i1 [[BOOL_EXPECTED]] to i32
// CHECK: [[SUB_RE:%.+]] = sub i32 [[EXPR_RE:%.+]], [[X_RVAL]]
// CHECK: [[SUB_IM:%.+]] = sub i32 [[EXPR_IM:%.+]], 0
// CHECK: icmp ne i32 [[SUB_RE]], 0
// CHECK: icmp ne i32 [[SUB_IM]], 0
// CHECK: [[BOOL_DESIRED:%.+]] = or i1
// CHECK: [[DESIRED:%.+]] = zext i1 [[BOOL_DESIRED]] to i8
// CHECK: store i8 [[DESIRED]], ptr [[TEMP:%.+]]
// CHECK: [[DESIRED:%.+]] = load i8, ptr [[TEMP]]
// CHECK: [[RES:%.+]] = cmpxchg ptr [[X_ADDR]], i8 [[EXPECTED]], i8 [[DESIRED]] monotonic monotonic, align 1
// CHECK: [[OLD_XI8:%.+]] = extractvalue { i8, i1 } [[RES]], 0
// CHECK: [[SUCCESS_FAIL:%.+]] = extractvalue { i8, i1 } [[RES]], 1
// CHECK: br i1 [[SUCCESS_FAIL]], label %[[EXIT:.+]], label %[[CONT]]
// CHECK: [[EXIT]]
#pragma omp atomic
  bx = civ - bx;
// CHECK: [[IDX:%.+]] = load i16, ptr @{{.+}}
// CHECK: load i8, ptr
// CHECK: [[VEC_ITEM_VAL:%.+]] = zext i1 %{{.+}} to i32
// CHECK: [[I128VAL:%.+]] = load atomic i128, ptr [[DEST:@.+]] monotonic, align 16
// CHECK: br label %[[CONT:.+]]
// CHECK: [[CONT]]
// CHECK: [[OLD_I128:%.+]] = phi i128 [ [[I128VAL]], %{{.+}} ], [ [[FAILED_I128_OLD_VAL:%.+]], %[[CONT]] ]
// CHECK: store i128 [[OLD_I128]], ptr [[TEMP:%.+]],
// CHECK: [[OLD_VEC_VAL:%.+]] = bitcast i128 [[OLD_I128]] to <4 x i32>
// CHECK: store <4 x i32> [[OLD_VEC_VAL]], ptr [[LDTEMP:%.+]],
// CHECK: [[VEC_VAL:%.+]] = load <4 x i32>, ptr [[LDTEMP]]
// CHECK: [[ITEM:%.+]] = extractelement <4 x i32> [[VEC_VAL]], i16 [[IDX]]
// CHECK: [[OR:%.+]] = or i32 [[ITEM]], [[VEC_ITEM_VAL]]
// CHECK: [[VEC_VAL:%.+]] = load <4 x i32>, ptr [[TEMP]]
// CHECK: [[NEW_VEC_VAL:%.+]] = insertelement <4 x i32> [[VEC_VAL]], i32 [[OR]], i16 [[IDX]]
// CHECK: store <4 x i32> [[NEW_VEC_VAL]], ptr [[TEMP]]
// CHECK: [[NEW_I128:%.+]] = load i128, ptr [[TEMP]]
// CHECK: [[RES:%.+]] = cmpxchg ptr [[DEST]], i128 [[OLD_I128]], i128 [[NEW_I128]] monotonic monotonic, align 16
// CHECK: [[FAILED_I128_OLD_VAL:%.+]] = extractvalue { i128, i1 } [[RES]], 0
// CHECK: [[FAIL_SUCCESS:%.+]] = extractvalue { i128, i1 } [[RES]], 1
// CHECK: br i1 [[FAIL_SUCCESS]], label %[[EXIT:.+]], label %[[CONT]]
// CHECK: [[EXIT]]
#pragma omp atomic update
  int4x[sv] |= bv;
// CHECK: [[EXPR:%.+]] = load x86_fp80, ptr @{{.+}}
// CHECK: [[PREV_VALUE:%.+]] = load atomic i32, ptr getelementptr (i8, ptr @{{.+}}, i64 4) monotonic, align 4
// CHECK: br label %[[CONT:.+]]
// CHECK: [[CONT]]
// CHECK: [[OLD_BF_VALUE:%.+]] = phi i32 [ [[PREV_VALUE]], %[[EXIT]] ], [ [[FAILED_OLD_VAL:%.+]], %[[CONT]] ]
// CHECK: store i32 [[OLD_BF_VALUE]], ptr [[TEMP1:%.+]],
// CHECK: store i32 [[OLD_BF_VALUE]], ptr [[TEMP:%.+]],
// CHECK: [[A_LD:%.+]] = load i32, ptr [[TEMP]],
// CHECK: [[A_SHL:%.+]] = shl i32 [[A_LD]], 1
// CHECK: [[A_ASHR:%.+]] = ashr i32 [[A_SHL]], 1
// CHECK: [[X_RVAL:%.+]] = sitofp i32 [[A_ASHR]] to x86_fp80
// CHECK: [[SUB:%.+]] = fsub x86_fp80 [[X_RVAL]], [[EXPR]]
// CHECK: [[CONV:%.+]] = fptosi x86_fp80 [[SUB]] to i32
// CHECK: [[NEW_VAL:%.+]] = load i32, ptr [[TEMP1]],
// CHECK: [[BF_VALUE:%.+]] = and i32 [[CONV]], 2147483647
// CHECK: [[BF_CLEAR:%.+]] = and i32 [[NEW_VAL]], -2147483648
// CHECK: or i32 [[BF_CLEAR]], [[BF_VALUE]]
// CHECK: store i32 %{{.+}}, ptr [[TEMP1]]
// CHECK: [[NEW_BF_VALUE:%.+]] = load i32, ptr [[TEMP1]]
// CHECK: [[RES:%.+]] = cmpxchg ptr getelementptr (i8, ptr @{{.+}}, i64 4), i32 [[OLD_BF_VALUE]], i32 [[NEW_BF_VALUE]] monotonic monotonic, align 4
// CHECK: [[FAILED_OLD_VAL]] = extractvalue { i32, i1 } [[RES]], 0
// CHECK: [[FAIL_SUCCESS:%.+]] = extractvalue { i32, i1 } [[RES]], 1
// CHECK: br i1 [[FAIL_SUCCESS]], label %[[EXIT:.+]], label %[[CONT]]
// CHECK: [[EXIT]]
#pragma omp atomic
  bfx.a = bfx.a - ldv;
// CHECK: [[EXPR:%.+]] = load x86_fp80, ptr @{{.+}}
// CHECK: call void @__atomic_load(i64 noundef 4, ptr noundef getelementptr (i8, ptr @{{.+}}, i64 4), ptr noundef [[LDTEMP:%.+]], i32 noundef 0)
// CHECK: br label %[[CONT:.+]]
// CHECK: [[CONT]]
// CHECK: [[PREV_VALUE:%.+]] = load i32, ptr [[LDTEMP]]
// CHECK: store i32 [[PREV_VALUE]], ptr [[TEMP1:%.+]],
// CHECK: [[PREV_VALUE:%.+]] = load i32, ptr [[LDTEMP]]
// CHECK: store i32 [[PREV_VALUE]], ptr [[TEMP:%.+]],
// CHECK: [[A_LD:%.+]] = load i32, ptr [[TEMP]],
// CHECK: [[A_SHL:%.+]] = shl i32 [[A_LD]], 1
// CHECK: [[A_ASHR:%.+]] = ashr i32 [[A_SHL]], 1
// CHECK: [[X_RVAL:%.+]] = sitofp i32 [[A_ASHR]] to x86_fp80
// CHECK: [[MUL:%.+]] = fmul x86_fp80 [[X_RVAL]], [[EXPR]]
// CHECK: [[CONV:%.+]] = fptosi x86_fp80 [[MUL]] to i32
// CHECK: [[NEW_VAL:%.+]] = load i32, ptr [[TEMP1]],
// CHECK: [[BF_VALUE:%.+]] = and i32 [[CONV]], 2147483647
// CHECK: [[BF_CLEAR:%.+]] = and i32 [[NEW_VAL]], -2147483648
// CHECK: or i32 [[BF_CLEAR]], [[BF_VALUE]]
// CHECK: store i32 %{{.+}}, ptr [[TEMP1]]
// CHECK: [[FAIL_SUCCESS:%.+]] = call zeroext i1 @__atomic_compare_exchange(i64 noundef 4, ptr noundef getelementptr (i8, ptr @{{.+}}, i64 4), ptr noundef [[LDTEMP]], ptr noundef [[TEMP1]], i32 noundef 0, i32 noundef 0)
// CHECK: br i1 [[FAIL_SUCCESS]], label %[[EXIT:.+]], label %[[CONT]]
// CHECK: [[EXIT]]
#pragma omp atomic update
  bfx_packed.a *= ldv;
// CHECK: [[EXPR:%.+]] = load x86_fp80, ptr @{{.+}}
// CHECK: [[PREV_VALUE:%.+]] = load atomic i32, ptr @{{.+}} monotonic, align 4
// CHECK: br label %[[CONT:.+]]
// CHECK: [[CONT]]
// CHECK: [[OLD_BF_VALUE:%.+]] = phi i32 [ [[PREV_VALUE]], %[[EXIT]] ], [ [[FAILED_OLD_VAL:%.+]], %[[CONT]] ]
// CHECK: store i32 [[OLD_BF_VALUE]], ptr [[TEMP1:%.+]],
// CHECK: store i32 [[OLD_BF_VALUE]], ptr [[TEMP:%.+]],
// CHECK: [[A_LD:%.+]] = load i32, ptr [[TEMP]],
// CHECK: [[A_ASHR:%.+]] = ashr i32 [[A_LD]], 31
// CHECK: [[X_RVAL:%.+]] = sitofp i32 [[A_ASHR]] to x86_fp80
// CHECK: [[SUB:%.+]] = fsub x86_fp80 [[X_RVAL]], [[EXPR]]
// CHECK: [[CONV:%.+]] = fptosi x86_fp80 [[SUB]] to i32
// CHECK: [[NEW_VAL:%.+]] = load i32, ptr [[TEMP1]],
// CHECK: [[BF_AND:%.+]] = and i32 [[CONV]], 1
// CHECK: [[BF_VALUE:%.+]] = shl i32 [[BF_AND]], 31
// CHECK: [[BF_CLEAR:%.+]] = and i32 [[NEW_VAL]], 2147483647
// CHECK: or i32 [[BF_CLEAR]], [[BF_VALUE]]
// CHECK: store i32 %{{.+}}, ptr [[TEMP1]]
// CHECK: [[NEW_BF_VALUE:%.+]] = load i32, ptr [[TEMP1]]
// CHECK: [[RES:%.+]] = cmpxchg ptr @{{.+}}, i32 [[OLD_BF_VALUE]], i32 [[NEW_BF_VALUE]] monotonic monotonic, align 4
// CHECK: [[FAILED_OLD_VAL]] = extractvalue { i32, i1 } [[RES]], 0
// CHECK: [[FAIL_SUCCESS:%.+]] = extractvalue { i32, i1 } [[RES]], 1
// CHECK: br i1 [[FAIL_SUCCESS]], label %[[EXIT:.+]], label %[[CONT]]
// CHECK: [[EXIT]]
#pragma omp atomic
  bfx2.a -= ldv;
// CHECK: [[EXPR:%.+]] = load x86_fp80, ptr @{{.+}}
// CHECK: [[PREV_VALUE:%.+]] = load atomic i8, ptr getelementptr (i8, ptr @{{.+}}, i64 3) monotonic, align 1
// CHECK: br label %[[CONT:.+]]
// CHECK: [[CONT]]
// CHECK: [[OLD_BF_VALUE:%.+]] = phi i8 [ [[PREV_VALUE]], %[[EXIT]] ], [ [[FAILED_OLD_VAL:%.+]], %[[CONT]] ]
// CHECK: store i8 [[OLD_BF_VALUE]], ptr [[BITCAST1:%.+]],
// CHECK: store i8 [[OLD_BF_VALUE]], ptr [[BITCAST:%.+]],
// CHECK: [[A_LD:%.+]] = load i8, ptr [[BITCAST]],
// CHECK: [[A_ASHR:%.+]] = ashr i8 [[A_LD]], 7
// CHECK: [[CAST:%.+]] = sext i8 [[A_ASHR]] to i32
// CHECK: [[X_RVAL:%.+]] = sitofp i32 [[CAST]] to x86_fp80
// CHECK: [[DIV:%.+]] = fdiv x86_fp80 [[EXPR]], [[X_RVAL]]
// CHECK: [[NEW_VAL:%.+]] = fptosi x86_fp80 [[DIV]] to i32
// CHECK: [[TRUNC:%.+]] = trunc i32 [[NEW_VAL]] to i8
// CHECK: [[BF_LD:%.+]] = load i8, ptr [[BITCAST1]],
// CHECK: [[BF_AND:%.+]] = and i8 [[TRUNC]], 1
// CHECK: [[BF_VALUE:%.+]] = shl i8 [[BF_AND]], 7
// CHECK: [[BF_CLEAR:%.+]] = and i8 %{{.+}}, 127
// CHECK: or i8 [[BF_CLEAR]], [[BF_VALUE]]
// CHECK: store i8 %{{.+}}, ptr [[BITCAST1]]
// CHECK: [[NEW_BF_VALUE:%.+]] = load i8, ptr [[BITCAST1]]
// CHECK: [[RES:%.+]] = cmpxchg ptr getelementptr (i8, ptr @{{.+}}, i64 3), i8 [[OLD_BF_VALUE]], i8 [[NEW_BF_VALUE]] monotonic monotonic, align 1
// CHECK: [[FAILED_OLD_VAL]] = extractvalue { i8, i1 } [[RES]], 0
// CHECK: [[FAIL_SUCCESS:%.+]] = extractvalue { i8, i1 } [[RES]], 1
// CHECK: br i1 [[FAIL_SUCCESS]], label %[[EXIT:.+]], label %[[CONT]]
// CHECK: [[EXIT]]
#pragma omp atomic update
  bfx2_packed.a = ldv / bfx2_packed.a;
// CHECK: [[EXPR:%.+]] = load x86_fp80, ptr @{{.+}}
// CHECK: [[PREV_VALUE:%.+]] = load atomic i32, ptr @{{.+}} monotonic, align 4
// CHECK: br label %[[CONT:.+]]
// CHECK: [[CONT]]
// CHECK: [[OLD_BF_VALUE:%.+]] = phi i32 [ [[PREV_VALUE]], %[[EXIT]] ], [ [[FAILED_OLD_VAL:%.+]], %[[CONT]] ]
// CHECK: store i32 [[OLD_BF_VALUE]], ptr [[TEMP1:%.+]],
// CHECK: store i32 [[OLD_BF_VALUE]], ptr [[TEMP:%.+]],
// CHECK: [[A_LD:%.+]] = load i32, ptr [[TEMP]],
// CHECK: [[A_SHL:%.+]] = shl i32 [[A_LD]], 7
// CHECK: [[A_ASHR:%.+]] = ashr i32 [[A_SHL]], 18
// CHECK: [[X_RVAL:%.+]] = sitofp i32 [[A_ASHR]] to x86_fp80
// CHECK: [[DIV:%.+]] = fdiv x86_fp80 [[X_RVAL]], [[EXPR]]
// CHECK: [[NEW_VAL:%.+]] = fptosi x86_fp80 [[DIV]] to i32
// CHECK: [[BF_LD:%.+]] = load i32, ptr [[TEMP1]],
// CHECK: [[BF_AND:%.+]] = and i32 [[NEW_VAL]], 16383
// CHECK: [[BF_VALUE:%.+]] = shl i32 [[BF_AND]], 11
// CHECK: [[BF_CLEAR:%.+]] = and i32 %{{.+}}, -33552385
// CHECK: or i32 [[BF_CLEAR]], [[BF_VALUE]]
// CHECK: store i32 %{{.+}}, ptr [[TEMP1]]
// CHECK: [[NEW_BF_VALUE:%.+]] = load i32, ptr [[TEMP1]]
// CHECK: [[RES:%.+]] = cmpxchg ptr @{{.+}}, i32 [[OLD_BF_VALUE]], i32 [[NEW_BF_VALUE]] monotonic monotonic, align 4
// CHECK: [[FAILED_OLD_VAL]] = extractvalue { i32, i1 } [[RES]], 0
// CHECK: [[FAIL_SUCCESS:%.+]] = extractvalue { i32, i1 } [[RES]], 1
// CHECK: br i1 [[FAIL_SUCCESS]], label %[[EXIT:.+]], label %[[CONT]]
// CHECK: [[EXIT]]
#pragma omp atomic
  bfx3.a /= ldv;
// CHECK: [[EXPR:%.+]] = load x86_fp80, ptr @{{.+}}
// CHECK: call void @__atomic_load(i64 noundef 3, ptr noundef getelementptr (i8, ptr @{{.+}}, i64 1), ptr noundef [[BITCAST:%.+]], i32 noundef 0)
// CHECK: br label %[[CONT:.+]]
// CHECK: [[CONT]]
// CHECK: [[PREV_VALUE:%.+]] = load i24, ptr [[LDTEMP:%.+]],
// CHECK: store i24 [[PREV_VALUE]], ptr [[TEMP1:%.+]],
// CHECK: [[PREV_VALUE:%.+]] = load i24, ptr [[LDTEMP]]
// CHECK: store i24 [[PREV_VALUE]], ptr [[TEMP:%.+]],
// CHECK: [[A_LD:%.+]] = load i24, ptr [[TEMP]],
// CHECK: [[A_SHL:%.+]] = shl i24 [[A_LD]], 7
// CHECK: [[A_ASHR:%.+]] = ashr i24 [[A_SHL]], 10
// CHECK: [[CAST:%.+]] = sext i24 [[A_ASHR]] to i32
// CHECK: [[X_RVAL:%.+]] = sitofp i32 [[CAST]] to x86_fp80
// CHECK: [[ADD:%.+]] = fadd x86_fp80 [[X_RVAL]], [[EXPR]]
// CHECK: [[NEW_VAL:%.+]] = fptosi x86_fp80 [[ADD]] to i32
// CHECK: [[TRUNC:%.+]] = trunc i32 [[NEW_VAL]] to i24
// CHECK: [[BF_LD:%.+]] = load i24, ptr [[TEMP1]],
// CHECK: [[BF_AND:%.+]] = and i24 [[TRUNC]], 16383
// CHECK: [[BF_VALUE:%.+]] = shl i24 [[BF_AND]], 3
// CHECK: [[BF_CLEAR:%.+]] = and i24 [[BF_LD]], -131065
// CHECK: or i24 [[BF_CLEAR]], [[BF_VALUE]]
// CHECK: store i24 %{{.+}}, ptr [[TEMP1]]
// CHECK: [[FAIL_SUCCESS:%.+]] = call zeroext i1 @__atomic_compare_exchange(i64 noundef 3, ptr noundef getelementptr (i8, ptr @{{.+}}, i64 1), ptr noundef [[LDTEMP]], ptr noundef [[TEMP1]], i32 noundef 0, i32 noundef 0)
// CHECK: br i1 [[FAIL_SUCCESS]], label %[[EXIT:.+]], label %[[CONT]]
// CHECK: [[EXIT]]
#pragma omp atomic update
  bfx3_packed.a += ldv;
// CHECK: [[EXPR:%.+]] = load x86_fp80, ptr @{{.+}}
// CHECK: [[PREV_VALUE:%.+]] = load atomic i64, ptr @{{.+}} monotonic, align 8
// CHECK: br label %[[CONT:.+]]
// CHECK: [[CONT]]
// CHECK: [[OLD_BF_VALUE:%.+]] = phi i64 [ [[PREV_VALUE]], %[[EXIT]] ], [ [[FAILED_OLD_VAL:%.+]], %[[CONT]] ]
// CHECK: store i64 [[OLD_BF_VALUE]], ptr [[TEMP1:%.+]],
// CHECK: store i64 [[OLD_BF_VALUE]], ptr [[TEMP:%.+]],
// CHECK: [[A_LD:%.+]] = load i64, ptr [[TEMP]],
// CHECK: [[A_SHL:%.+]] = shl i64 [[A_LD]], 47
// CHECK: [[A_ASHR:%.+]] = ashr i64 [[A_SHL:%.+]], 63
// CHECK: [[A_CAST:%.+]] = trunc i64 [[A_ASHR:%.+]] to i32
// CHECK: [[X_RVAL:%.+]] = sitofp i32 [[CAST:%.+]] to x86_fp80
// CHECK: [[MUL:%.+]] = fmul x86_fp80 [[X_RVAL]], [[EXPR]]
// CHECK: [[NEW_VAL:%.+]] = fptosi x86_fp80 [[MUL]] to i32
// CHECK: [[ZEXT:%.+]] = zext i32 [[NEW_VAL]] to i64
// CHECK: [[BF_LD:%.+]] = load i64, ptr [[TEMP1]],
// CHECK: [[BF_AND:%.+]] = and i64 [[ZEXT]], 1
// CHECK: [[BF_VALUE:%.+]] = shl i64 [[BF_AND]], 16
// CHECK: [[BF_CLEAR:%.+]] = and i64 [[BF_LD]], -65537
// CHECK: or i64 [[BF_CLEAR]], [[BF_VALUE]]
// CHECK: store i64 %{{.+}}, ptr [[TEMP1]]
// CHECK: [[NEW_BF_VALUE:%.+]] = load i64, ptr [[TEMP1]]
// CHECK: [[RES:%.+]] = cmpxchg ptr @{{.+}}, i64 [[OLD_BF_VALUE]], i64 [[NEW_BF_VALUE]] monotonic monotonic, align 8
// CHECK: [[FAILED_OLD_VAL]] = extractvalue { i64, i1 } [[RES]], 0
// CHECK: [[FAIL_SUCCESS:%.+]] = extractvalue { i64, i1 } [[RES]], 1
// CHECK: br i1 [[FAIL_SUCCESS]], label %[[EXIT:.+]], label %[[CONT]]
// CHECK: [[EXIT]]
#pragma omp atomic
  bfx4.a = bfx4.a * ldv;
// CHECK: [[EXPR:%.+]] = load x86_fp80, ptr @{{.+}}
// CHECK: [[PREV_VALUE:%.+]] = load atomic i8, ptr getelementptr inbounds (%struct.BitFields4_packed, ptr @{{.+}}, i32 0, i32 1) monotonic, align 1
// CHECK: br label %[[CONT:.+]]
// CHECK: [[CONT]]
// CHECK: [[OLD_BF_VALUE:%.+]] = phi i8 [ [[PREV_VALUE]], %{{.+}} ], [ [[FAILED_OLD_VAL:%.+]], %[[CONT]] ]
// CHECK: store i8 [[OLD_BF_VALUE]], ptr [[BITCAST1:%.+]],
// CHECK: store i8 [[OLD_BF_VALUE]], ptr [[BITCAST:%.+]],
// CHECK: [[A_LD:%.+]] = load i8, ptr [[BITCAST]],
// CHECK: [[A_SHL:%.+]] = shl i8 [[A_LD]], 7
// CHECK: [[A_ASHR:%.+]] = ashr i8 [[A_SHL:%.+]], 7
// CHECK: [[CAST:%.+]] = sext i8 [[A_ASHR:%.+]] to i32
// CHECK: [[CONV:%.+]] = sitofp i32 [[CAST]] to x86_fp80
// CHECK: [[SUB: %.+]] = fsub x86_fp80 [[CONV]], [[EXPR]]
// CHECK: [[CONV:%.+]] = fptosi x86_fp80 [[SUB:%.+]] to i32
// CHECK: [[NEW_VAL:%.+]] = trunc i32 [[CONV]] to i8
// CHECK: [[BF_LD:%.+]] = load i8, ptr [[BITCAST1]],
// CHECK: [[BF_VALUE:%.+]] = and i8 [[NEW_VAL]], 1
// CHECK: [[BF_CLEAR:%.+]] = and i8 [[BF_LD]], -2
// CHECK: or i8 [[BF_CLEAR]], [[BF_VALUE]]
// CHECK: store i8 %{{.+}}, ptr [[BITCAST1]]
// CHECK: [[NEW_BF_VALUE:%.+]] = load i8, ptr [[BITCAST1]]
// CHECK: [[RES:%.+]] = cmpxchg ptr getelementptr inbounds (%struct.BitFields4_packed, ptr @{{.+}}, i32 0, i32 1), i8 [[OLD_BF_VALUE]], i8 [[NEW_BF_VALUE]] monotonic monotonic, align 1
// CHECK: [[FAILED_OLD_VAL]] = extractvalue { i8, i1 } [[RES]], 0
// CHECK: [[FAIL_SUCCESS:%.+]] = extractvalue { i8, i1 } [[RES]], 1
// CHECK: br i1 [[FAIL_SUCCESS]], label %[[EXIT:.+]], label %[[CONT]]
// CHECK: [[EXIT]]
#pragma omp atomic relaxed update
  bfx4_packed.a -= ldv;
// CHECK: [[EXPR:%.+]] = load x86_fp80, ptr @{{.+}}
// CHECK: [[PREV_VALUE:%.+]] = load atomic i64, ptr @{{.+}} monotonic, align 8
// CHECK: br label %[[CONT:.+]]
// CHECK: [[CONT]]
// CHECK: [[OLD_BF_VALUE:%.+]] = phi i64 [ [[PREV_VALUE]], %[[EXIT]] ], [ [[FAILED_OLD_VAL:%.+]], %[[CONT]] ]
// CHECK: store i64 [[OLD_BF_VALUE]], ptr [[TEMP1:%.+]],
// CHECK: store i64 [[OLD_BF_VALUE]], ptr [[TEMP:%.+]],
// CHECK: [[A_LD:%.+]] = load i64, ptr [[TEMP]],
// CHECK: [[A_SHL:%.+]] = shl i64 [[A_LD]], 40
// CHECK: [[A_ASHR:%.+]] = ashr i64 [[A_SHL:%.+]], 57
// CHECK: [[CONV:%.+]] = sitofp i64 [[A_ASHR]] to x86_fp80
// CHECK: [[DIV:%.+]] = fdiv x86_fp80 [[CONV]], [[EXPR]]
// CHECK: [[CONV:%.+]] = fptosi x86_fp80 [[DIV]] to i64
// CHECK: [[BF_LD:%.+]] = load i64, ptr [[TEMP1]],
// CHECK: [[BF_AND:%.+]] = and i64 [[CONV]], 127
// CHECK: [[BF_VALUE:%.+]] = shl i64 [[BF_AND:%.+]], 17
// CHECK: [[BF_CLEAR:%.+]] = and i64 [[BF_LD]], -16646145
// CHECK: [[VAL:%.+]] = or i64 [[BF_CLEAR]], [[BF_VALUE]]
// CHECK: store i64 [[VAL]], ptr [[TEMP1]]
// CHECK: [[NEW_BF_VALUE:%.+]] = load i64, ptr [[TEMP1]]
// CHECK: [[RES:%.+]] = cmpxchg ptr @{{.+}}, i64 [[OLD_BF_VALUE]], i64 [[NEW_BF_VALUE]] monotonic monotonic, align 8
// CHECK: [[FAILED_OLD_VAL]] = extractvalue { i64, i1 } [[RES]], 0
// CHECK: [[FAIL_SUCCESS:%.+]] = extractvalue { i64, i1 } [[RES]], 1
// CHECK: br i1 [[FAIL_SUCCESS]], label %[[EXIT:.+]], label %[[CONT]]
// CHECK: [[EXIT]]
#pragma omp atomic
  bfx4.b /= ldv;
// CHECK: [[EXPR:%.+]] = load x86_fp80, ptr @{{.+}}
// CHECK: [[PREV_VALUE:%.+]] = load atomic i8, ptr getelementptr inbounds (%struct.BitFields4_packed, ptr @{{.+}}, i32 0, i32 1) monotonic, align 1
// CHECK: br label %[[CONT:.+]]
// CHECK: [[CONT]]
// CHECK: [[OLD_BF_VALUE:%.+]] = phi i8 [ [[PREV_VALUE]], %[[EXIT]] ], [ [[FAILED_OLD_VAL:%.+]], %[[CONT]] ]
// CHECK: store i8 [[OLD_BF_VALUE]], ptr [[BITCAST1:%.+]],
// CHECK: store i8 [[OLD_BF_VALUE]], ptr [[BITCAST:%.+]],
// CHECK: [[A_LD:%.+]] = load i8, ptr [[BITCAST]],
// CHECK: [[A_ASHR:%.+]] = ashr i8 [[A_LD]], 1
// CHECK: [[CAST:%.+]] = sext i8 [[A_ASHR]] to i64
// CHECK: [[CONV:%.+]] = sitofp i64 [[CAST]] to x86_fp80
// CHECK: [[ADD:%.+]] = fadd x86_fp80 [[CONV]], [[EXPR]]
// CHECK: [[NEW_VAL:%.+]] = fptosi x86_fp80 [[ADD]] to i64
// CHECK: [[TRUNC:%.+]] = trunc i64 [[NEW_VAL]] to i8
// CHECK: [[BF_LD:%.+]] = load i8, ptr [[BITCAST1]],
// CHECK: [[BF_AND:%.+]] = and i8 [[TRUNC]], 127
// CHECK: [[BF_VALUE:%.+]] = shl i8 [[BF_AND]], 1
// CHECK: [[BF_CLEAR:%.+]] = and i8 [[BF_LD]], 1
// CHECK: or i8 [[BF_CLEAR]], [[BF_VALUE]]
// CHECK: store i8 %{{.+}}, ptr [[BITCAST1]]
// CHECK: [[NEW_BF_VALUE:%.+]] = load i8, ptr [[BITCAST1]]
// CHECK: [[RES:%.+]] = cmpxchg ptr getelementptr inbounds (%struct.BitFields4_packed, ptr @{{.+}}, i32 0, i32 1), i8 [[OLD_BF_VALUE]], i8 [[NEW_BF_VALUE]] monotonic monotonic, align 1
// CHECK: [[FAILED_OLD_VAL]] = extractvalue { i8, i1 } [[RES]], 0
// CHECK: [[FAIL_SUCCESS:%.+]] = extractvalue { i8, i1 } [[RES]], 1
// CHECK: br i1 [[FAIL_SUCCESS]], label %[[EXIT:.+]], label %[[CONT]]
// CHECK: [[EXIT]]
#pragma omp atomic update relaxed
  bfx4_packed.b += ldv;
// CHECK: load i64, ptr
// CHECK: [[EXPR:%.+]] = uitofp i64 %{{.+}} to float
// CHECK: [[I64VAL:%.+]] = load atomic i64, ptr [[DEST:@.+]] monotonic, align 8
// CHECK: br label %[[CONT:.+]]
// CHECK: [[CONT]]
// CHECK: [[OLD_I64:%.+]] = phi i64 [ [[I64VAL]], %{{.+}} ], [ [[FAILED_I64_OLD_VAL:%.+]], %[[CONT]] ]
// CHECK: store i64 [[OLD_I64]], ptr [[TEMP:%.+]],
// CHECK: [[OLD_VEC_VAL:%.+]] = bitcast i64 [[OLD_I64]] to <2 x float>
// CHECK: store <2 x float> [[OLD_VEC_VAL]], ptr [[LDTEMP:%.+]],
// CHECK: [[VEC_VAL:%.+]] = load <2 x float>, ptr [[LDTEMP]]
// CHECK: [[X:%.+]] = extractelement <2 x float> [[VEC_VAL]], i64 0
// CHECK: [[VEC_ITEM_VAL:%.+]] = fsub float [[EXPR]], [[X]]
// CHECK: [[VEC_VAL:%.+]] = load <2 x float>, ptr [[TEMP]],
// CHECK: [[NEW_VEC_VAL:%.+]] = insertelement <2 x float> [[VEC_VAL]], float [[VEC_ITEM_VAL]], i64 0
// CHECK: store <2 x float> [[NEW_VEC_VAL]], ptr [[TEMP]]
// CHECK: [[NEW_I64:%.+]] = load i64, ptr [[TEMP]]
// CHECK: [[RES:%.+]] = cmpxchg ptr [[DEST]], i64 [[OLD_I64]], i64 [[NEW_I64]] monotonic monotonic, align 8
// CHECK: [[FAILED_I64_OLD_VAL:%.+]] = extractvalue { i64, i1 } [[RES]], 0
// CHECK: [[FAIL_SUCCESS:%.+]] = extractvalue { i64, i1 } [[RES]], 1
// CHECK: br i1 [[FAIL_SUCCESS]], label %[[EXIT:.+]], label %[[CONT]]
// CHECK: [[EXIT]]
#pragma omp atomic relaxed
  float2x.x = ulv - float2x.x;
// CHECK: [[EXPR:%.+]] = load double, ptr @{{.+}},
// CHECK: [[OLD_VAL:%.+]] = call i32 @llvm.read_register.i32([[REG:metadata ![0-9]+]])
// CHECK: [[X_RVAL:%.+]] = sitofp i32 [[OLD_VAL]] to double
// CHECK: [[DIV:%.+]] = fdiv double [[EXPR]], [[X_RVAL]]
// CHECK: [[NEW_VAL:%.+]] = fptosi double [[DIV]] to i32
// CHECK: call void @llvm.write_register.i32([[REG]], i32 [[NEW_VAL]])
// CHECK: call{{.*}} @__kmpc_flush(
#pragma omp atomic seq_cst
  rix = dv / rix;
  return 0;
}

#endif

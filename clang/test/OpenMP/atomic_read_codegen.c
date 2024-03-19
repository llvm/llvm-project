// RUN: %clang_cc1 -verify -triple x86_64-apple-darwin10 -target-cpu core2 -fopenmp -x c -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -fopenmp -x c -triple x86_64-apple-darwin10 -target-cpu core2 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -x c -triple x86_64-apple-darwin10 -target-cpu core2 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s

// RUN: %clang_cc1 -verify -triple x86_64-apple-darwin10 -target-cpu core2 -fopenmp-simd -x c -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -fopenmp-simd -x c -triple x86_64-apple-darwin10 -target-cpu core2 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -x c -triple x86_64-apple-darwin10 -target-cpu core2 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// SIMD-ONLY0-NOT: {{__kmpc|__tgt}}
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

// CHECK-LABEL: @main(
int main(void) {
// CHECK: load atomic i8, ptr {{.*}} monotonic, align 1
// CHECK: store i8
#pragma omp atomic read
  bv = bx;
// CHECK: load atomic i8, ptr {{.*}} monotonic, align 1
// CHECK: store i8
#pragma omp atomic read
  cv = cx;
// CHECK: load atomic i8, ptr {{.*}} monotonic, align 1
// CHECK: store i8
#pragma omp atomic read
  ucv = ucx;
// CHECK: load atomic i16, ptr {{.*}} monotonic, align 2
// CHECK: store i16
#pragma omp atomic read
  sv = sx;
// CHECK: load atomic i16, ptr {{.*}} monotonic, align 2
// CHECK: store i16
#pragma omp atomic read
  usv = usx;
// CHECK: load atomic i32, ptr {{.*}} monotonic, align 4
// CHECK: store i32
#pragma omp atomic read
  iv = ix;
// CHECK: load atomic i32, ptr {{.*}} monotonic, align 4
// CHECK: store i32
#pragma omp atomic read
  uiv = uix;
// CHECK: load atomic i64, ptr {{.*}} monotonic, align 8
// CHECK: store i64
#pragma omp atomic read
  lv = lx;
// CHECK: load atomic i64, ptr {{.*}} monotonic, align 8
// CHECK: store i64
#pragma omp atomic read
  ulv = ulx;
// CHECK: load atomic i64, ptr {{.*}} monotonic, align 8
// CHECK: store i64
#pragma omp atomic read
  llv = llx;
// CHECK: load atomic i64, ptr {{.*}} monotonic, align 8
// CHECK: store i64
#pragma omp atomic read
  ullv = ullx;
// CHECK: load atomic float, ptr {{.*}} monotonic, align 4
// CHECK: store float
#pragma omp atomic read
  fv = fx;
// CHECK: load atomic double, ptr {{.*}} monotonic, align 8
// CHECK: store double
#pragma omp atomic read
  dv = dx;
// CHECK: [[LD:%.+]] = load atomic i128, ptr {{.*}} monotonic, align 16
// CHECK: store i128 [[LD]], ptr [[LDTEMP:%.+]]
// CHECK: [[LD:%.+]] = load x86_fp80, ptr [[LDTEMP]]
// CHECK: store x86_fp80 [[LD]]
#pragma omp atomic read
  ldv = ldx;
// CHECK: call{{.*}} void @__atomic_load(i64 noundef 8,
// CHECK: store i32
// CHECK: store i32
#pragma omp atomic read
  civ = cix;
// CHECK: call{{.*}} void @__atomic_load(i64 noundef 8,
// CHECK: store float
// CHECK: store float
#pragma omp atomic read
  cfv = cfx;
// CHECK: call{{.*}} void @__atomic_load(i64 noundef 16,
// CHECK: call{{.*}} @__kmpc_flush(
// CHECK: store double
// CHECK: store double
#pragma omp atomic seq_cst read
  cdv = cdx;
// CHECK: load atomic i64, ptr {{.*}} monotonic, align 8
// CHECK: store i8
#pragma omp atomic read
  bv = ulx;
// CHECK: load atomic i8, ptr {{.*}} monotonic, align 1
// CHECK: store i8
#pragma omp atomic read
  cv = bx;
// CHECK: load atomic i8, ptr {{.*}} seq_cst, align 1
// CHECK: call{{.*}} @__kmpc_flush(
// CHECK: store i8
#pragma omp atomic read seq_cst
  ucv = cx;
// CHECK: load atomic i64, ptr {{.*}} monotonic, align 8
// CHECK: store i16
#pragma omp atomic read
  sv = ulx;
// CHECK: load atomic i64, ptr {{.*}} monotonic, align 8
// CHECK: store i16
#pragma omp atomic read
  usv = lx;
// CHECK: load atomic i32, ptr {{.*}} seq_cst, align 4
// CHECK: call{{.*}} @__kmpc_flush(
// CHECK: store i32
#pragma omp atomic seq_cst, read
  iv = uix;
// CHECK: load atomic i32, ptr {{.*}} monotonic, align 4
// CHECK: store i32
#pragma omp atomic read
  uiv = ix;
// CHECK: call{{.*}} void @__atomic_load(i64 noundef 8,
// CHECK: store i64
#pragma omp atomic read
  lv = cix;
// CHECK: load atomic float, ptr {{.*}} monotonic, align 4
// CHECK: store i64
#pragma omp atomic read
  ulv = fx;
// CHECK: load atomic double, ptr {{.*}} monotonic, align 8
// CHECK: store i64
#pragma omp atomic read
  llv = dx;
// CHECK: load atomic i128, ptr {{.*}} monotonic, align 16
// CHECK: store i64
#pragma omp atomic read
  ullv = ldx;
// CHECK: call{{.*}} void @__atomic_load(i64 noundef 8,
// CHECK: store float
#pragma omp atomic read
  fv = cix;
// CHECK: load atomic i16, ptr {{.*}} monotonic, align 2
// CHECK: store double
#pragma omp atomic read
  dv = sx;
// CHECK: load atomic i8, ptr {{.*}} monotonic, align 1
// CHECK: store x86_fp80
#pragma omp atomic read
  ldv = bx;
// CHECK: load atomic i8, ptr {{.*}} monotonic, align 1
// CHECK: store i32
// CHECK: store i32
#pragma omp atomic read
  civ = bx;
// CHECK: load atomic i16, ptr {{.*}} monotonic, align 2
// CHECK: store float
// CHECK: store float
#pragma omp atomic read
  cfv = usx;
// CHECK: load atomic i64, ptr {{.*}} monotonic, align 8
// CHECK: store double
// CHECK: store double
#pragma omp atomic read
  cdv = llx;
// CHECK: [[I128VAL:%.+]] = load atomic i128, ptr @{{.+}} monotonic, align 16
// CHECK: store i128 [[I128VAL]], ptr [[LDTEMP:%.+]]
// CHECK: [[LD:%.+]] = load <4 x i32>, ptr [[LDTEMP]]
// CHECK: extractelement <4 x i32> [[LD]]
// CHECK: store i8
#pragma omp atomic read
  bv = int4x[0];
// CHECK: [[LD:%.+]] = load atomic i32, ptr getelementptr (i8, ptr @{{.+}}, i64 4) monotonic, align 4
// CHECK: store i32 [[LD]], ptr [[LDTEMP:%.+]]
// CHECK: [[LD:%.+]] = load i32, ptr [[LDTEMP]]
// CHECK: [[SHL:%.+]] = shl i32 [[LD]], 1
// CHECK: ashr i32 [[SHL]], 1
// CHECK: store x86_fp80
#pragma omp atomic read
  ldv = bfx.a;
// CHECK: call void @__atomic_load(i64 noundef 4, ptr noundef getelementptr (i8, ptr @bfx_packed, i64 4), ptr noundef [[LDTEMP:%.+]], i32 noundef 0)
// CHECK: [[LD:%.+]] = load i32, ptr [[LDTEMP]]
// CHECK: [[SHL:%.+]] = shl i32 [[LD]], 1
// CHECK: ashr i32 [[SHL]], 1
// CHECK: store x86_fp80
#pragma omp atomic read
  ldv = bfx_packed.a;
// CHECK: [[LD:%.+]] = load atomic i32, ptr @bfx2 monotonic, align 4
// CHECK: store i32 [[LD]], ptr [[LDTEMP:%.+]]
// CHECK: [[LD:%.+]] = load i32, ptr [[LDTEMP]]
// CHECK: ashr i32 [[LD]], 31
// CHECK: store x86_fp80
#pragma omp atomic read
  ldv = bfx2.a;
// CHECK: [[LD:%.+]] = load atomic i8, ptr getelementptr (i8, ptr @bfx2_packed, i64 3) monotonic, align 1
// CHECK: store i8 [[LD]], ptr [[LDTEMP:%.+]]
// CHECK: [[LD:%.+]] = load i8, ptr [[LDTEMP]]
// CHECK: ashr i8 [[LD]], 7
// CHECK: store x86_fp80
#pragma omp atomic read
  ldv = bfx2_packed.a;
// CHECK: [[LD:%.+]] = load atomic i32, ptr @bfx3 monotonic, align 4
// CHECK: store i32 [[LD]], ptr [[LDTEMP:%.+]]
// CHECK: [[LD:%.+]] = load i32, ptr [[LDTEMP]]
// CHECK: [[SHL:%.+]] = shl i32 [[LD]], 7
// CHECK: ashr i32 [[SHL]], 18
// CHECK: store x86_fp80
#pragma omp atomic read
  ldv = bfx3.a;
// CHECK: call void @__atomic_load(i64 noundef 3, ptr noundef getelementptr (i8, ptr @bfx3_packed, i64 1), ptr noundef [[LDTEMP:%.+]], i32 noundef 0)
// CHECK: [[LD:%.+]] = load i24, ptr [[LDTEMP]]
// CHECK: [[SHL:%.+]] = shl i24 [[LD]], 7
// CHECK: [[ASHR:%.+]] = ashr i24 [[SHL]], 10
// CHECK: sext i24 [[ASHR]] to i32
// CHECK: store x86_fp80
#pragma omp atomic read
  ldv = bfx3_packed.a;
// CHECK: [[LD:%.+]] = load atomic i64, ptr @bfx4 monotonic, align 8
// CHECK: store i64 [[LD]], ptr [[LDTEMP:%.+]]
// CHECK: [[LD:%.+]] = load i64, ptr [[LDTEMP]]
// CHECK: [[SHL:%.+]] = shl i64 [[LD]], 47
// CHECK: [[ASHR:%.+]] = ashr i64 [[SHL]], 63
// CHECK: trunc i64 [[ASHR]] to i32
// CHECK: store x86_fp80
#pragma omp atomic read
  ldv = bfx4.a;
// CHECK: [[LD:%.+]] = load atomic i8, ptr getelementptr (i8, ptr @bfx4_packed, i64 2) monotonic, align 1
// CHECK: store i8 [[LD]], ptr [[LDTEMP:%.+]]
// CHECK: [[LD:%.+]] = load i8, ptr [[LDTEMP]]
// CHECK: [[SHL:%.+]] = shl i8 [[LD]], 7
// CHECK: [[ASHR:%.+]] = ashr i8 [[SHL]], 7
// CHECK: sext i8 [[ASHR]] to i32
// CHECK: store x86_fp80
#pragma omp atomic relaxed read
  ldv = bfx4_packed.a;
// CHECK: [[LD:%.+]] = load atomic i64, ptr @bfx4 monotonic, align 8
// CHECK: store i64 [[LD]], ptr [[LDTEMP:%.+]]
// CHECK: [[LD:%.+]] = load i64, ptr [[LDTEMP]]
// CHECK: [[SHL:%.+]] = shl i64 [[LD]], 40
// CHECK: [[ASHR:%.+]] = ashr i64 [[SHL]], 57
// CHECK: store x86_fp80
#pragma omp atomic read relaxed
  ldv = bfx4.b;
// CHECK: [[LD:%.+]] = load atomic i8, ptr getelementptr (i8, ptr @bfx4_packed, i64 2) acquire, align 1
// CHECK: store i8 [[LD]], ptr [[LDTEMP:%.+]]
// CHECK: [[LD:%.+]] = load i8, ptr [[LDTEMP]]
// CHECK: [[ASHR:%.+]] = ashr i8 [[LD]], 1
// CHECK: sext i8 [[ASHR]] to i64
// CHECK: call{{.*}} @__kmpc_flush(
// CHECK: store x86_fp80
#pragma omp atomic read acquire
  ldv = bfx4_packed.b;
// CHECK: [[LD:%.+]] = load atomic i64, ptr @{{.+}} monotonic, align 8
// CHECK: store i64 [[LD]], ptr [[LDTEMP:%.+]]
// CHECK: [[LD:%.+]] = load <2 x float>, ptr [[LDTEMP]]
// CHECK: extractelement <2 x float> [[LD]]
// CHECK: store i64
#pragma omp atomic read
  ulv = float2x.x;
// CHECK: call{{.*}} i{{[0-9]+}} @llvm.read_register
// CHECK: call{{.*}} @__kmpc_flush(
// CHECK: store double
#pragma omp atomic read seq_cst
  dv = rix;
  return 0;
}

#endif

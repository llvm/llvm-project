// RUN: %clang_cc1 -triple powerpc64le-unknown-linux-gnu -emit-llvm \
// RUN:   -target-cpu pwr9 -target-feature +float128 -mabi=ieeelongdouble \
// RUN:   -o - %s | FileCheck %s -check-prefix=IEEE
// RUN: %clang_cc1 -triple powerpc64le-unknown-linux-gnu -emit-llvm \
// RUN:   -target-cpu pwr9 -target-feature +float128 \
// RUN:   -o - %s | FileCheck %s -check-prefix=IBM

// RUN: %clang -target powerpc64le-unknown-linux-gnu -S -emit-llvm \
// RUN:   -fopenmp-targets=ppc64le -mfloat128 -mabi=ieeelongdouble -mcpu=pwr9 \
// RUN:   -Xopenmp-target=ppc64le -mcpu=pwr9 -Xopenmp-target=ppc64le \
// RUN:   -mfloat128 -fopenmp=libomp -o - %s | FileCheck %s -check-prefix=OMP

#include <stdarg.h>

void foo_ld(long double);
void foo_fq(__float128);

// Verify cases when OpenMP target's and host's long-double semantics differ.

// OMP-LABEL: define internal void @.omp_outlined.
// OMP: %[[CUR:[0-9a-zA-Z_.]+]] = load i8*, i8**
// OMP: %[[V2:[0-9a-zA-Z_.]+]] = bitcast i8* %[[CUR]] to ppc_fp128*
// OMP: %[[V3:[0-9a-zA-Z_.]+]] = load ppc_fp128, ppc_fp128* %[[V2]], align 8
// OMP: call void @foo_ld(ppc_fp128 %[[V3]])

// OMP-LABEL: define dso_local void @omp
// OMP: %[[AP1:[0-9a-zA-Z_.]+]] = bitcast i8** %[[AP:[0-9a-zA-Z_.]+]] to i8*
// OMP: call void @llvm.va_start(i8* %[[AP1]])
// OMP: %[[CUR:[0-9a-zA-Z_.]+]] = load i8*, i8** %[[AP]], align 8
// OMP: %[[V0:[0-9a-zA-Z_.]+]] = ptrtoint i8* %[[CUR]] to i64
// OMP: %[[V1:[0-9a-zA-Z_.]+]] = add i64 %[[V0]], 15
// OMP: %[[V2:[0-9a-zA-Z_.]+]] = and i64 %[[V1]], -16
// OMP: %[[ALIGN:[0-9a-zA-Z_.]+]] = inttoptr i64 %[[V2]] to i8*
// OMP: %[[V3:[0-9a-zA-Z_.]+]] = bitcast i8* %[[ALIGN]] to fp128*
// OMP: %[[V4:[0-9a-zA-Z_.]+]] = load fp128, fp128* %[[V3]], align 16
// OMP: call void @foo_ld(fp128 %[[V4]])
void omp(int n, ...) {
  va_list ap;
  va_start(ap, n);
  foo_ld(va_arg(ap, long double));
  #pragma omp target parallel
  for (int i = 1; i < n; ++i) {
    foo_ld(va_arg(ap, long double));
  }
  va_end(ap);
}

// IEEE-LABEL: define void @f128
// IEEE: %[[AP1:[0-9a-zA-Z_.]+]] = bitcast i8** %[[AP:[0-9a-zA-Z_.]+]] to i8*
// IEEE: call void @llvm.va_start(i8* %[[AP1]])
// IEEE: %[[CUR:[0-9a-zA-Z_.]+]] = load i8*, i8** %[[AP]]
// IEEE: %[[V0:[0-9a-zA-Z_.]+]] = ptrtoint i8* %[[CUR]] to i64
// IEEE: %[[V1:[0-9a-zA-Z_.]+]] = add i64 %[[V0]], 15
// IEEE: %[[V2:[0-9a-zA-Z_.]+]] = and i64 %[[V1]], -16
// IEEE: %[[ALIGN:[0-9a-zA-Z_.]+]] = inttoptr i64 %[[V2]] to i8*
// IEEE: %[[V3:[0-9a-zA-Z_.]+]] = bitcast i8* %[[ALIGN]] to fp128*
// IEEE: %[[V4:[0-9a-zA-Z_.]+]] = load fp128, fp128* %[[V3]], align 16
// IEEE: call void @foo_fq(fp128 %[[V4]])
// IEEE: %[[AP2:[0-9a-zA-Z_.]+]] = bitcast i8** %[[AP]] to i8*
// IEEE: call void @llvm.va_end(i8* %[[AP2]])
void f128(int n, ...) {
  va_list ap;
  va_start(ap, n);
  foo_fq(va_arg(ap, __float128));
  va_end(ap);
}

// IEEE-LABEL: define void @long_double
// IEEE: %[[AP1:[0-9a-zA-Z_.]+]] = bitcast i8** %[[AP:[0-9a-zA-Z_.]+]] to i8*
// IEEE: call void @llvm.va_start(i8* %[[AP1]])
// IEEE: %[[CUR:[0-9a-zA-Z_.]+]] = load i8*, i8** %[[AP]]
// IEEE: %[[V0:[0-9a-zA-Z_.]+]] = ptrtoint i8* %[[CUR]] to i64
// IEEE: %[[V1:[0-9a-zA-Z_.]+]] = add i64 %[[V0]], 15
// IEEE: %[[V2:[0-9a-zA-Z_.]+]] = and i64 %[[V1]], -16
// IEEE: %[[ALIGN:[0-9a-zA-Z_.]+]] = inttoptr i64 %[[V2]] to i8*
// IEEE: %[[V3:[0-9a-zA-Z_.]+]] = bitcast i8* %[[ALIGN]] to fp128*
// IEEE: %[[V4:[0-9a-zA-Z_.]+]] = load fp128, fp128* %[[V3]], align 16
// IEEE: call void @foo_ld(fp128 %[[V4]])
// IEEE: %[[AP2:[0-9a-zA-Z_.]+]] = bitcast i8** %[[AP]] to i8*
// IEEE: call void @llvm.va_end(i8* %[[AP2]])

// IBM-LABEL: define void @long_double
// IBM: %[[AP1:[0-9a-zA-Z_.]+]] = bitcast i8** %[[AP:[0-9a-zA-Z_.]+]] to i8*
// IBM: call void @llvm.va_start(i8* %[[AP1]])
// IBM: %[[CUR:[0-9a-zA-Z_.]+]] = load i8*, i8** %[[AP]]
// IBM: %[[V3:[0-9a-zA-Z_.]+]] = bitcast i8* %[[CUR]] to ppc_fp128*
// IBM: %[[V4:[0-9a-zA-Z_.]+]] = load ppc_fp128, ppc_fp128* %[[V3]], align 8
// IBM: call void @foo_ld(ppc_fp128 %[[V4]])
// IBM: %[[AP2:[0-9a-zA-Z_.]+]] = bitcast i8** %[[AP]] to i8*
// IBM: call void @llvm.va_end(i8* %[[AP2]])
void long_double(int n, ...) {
  va_list ap;
  va_start(ap, n);
  foo_ld(va_arg(ap, long double));
  va_end(ap);
}

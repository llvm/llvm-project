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

// RUN: %clang_cc1 -no-opaque-pointers -triple powerpc64le-unknown-linux-gnu -emit-llvm \
// RUN:   -target-cpu pwr9 -target-feature +float128 -mabi=ieeelongdouble \
// RUN:   -o - %s | FileCheck %s -check-prefix=IEEE
// RUN: %clang_cc1 -no-opaque-pointers -triple powerpc64le-unknown-linux-gnu -emit-llvm \
// RUN:   -target-cpu pwr9 -target-feature +float128 \
// RUN:   -o - %s | FileCheck %s -check-prefix=IBM

// RUN: %clang_cc1 -no-opaque-pointers -triple ppc64le -emit-llvm-bc %s -target-cpu pwr9 \
// RUN:   -target-feature +float128 -mabi=ieeelongdouble -fopenmp \
// RUN:   -fopenmp-targets=ppc64le -o %t-ppc-host.bc
// RUN: %clang_cc1 -no-opaque-pointers -triple ppc64le -aux-triple ppc64le %s -target-cpu pwr9 \
// RUN:   -target-feature +float128 -fopenmp -fopenmp-is-device -emit-llvm \
// RUN:   -fopenmp-host-ir-file-path %t-ppc-host.bc -o - | FileCheck %s \
// RUN:   -check-prefix=OMP-TARGET
// RUN: %clang_cc1 -no-opaque-pointers -triple ppc64le %t-ppc-host.bc -emit-llvm -o - | FileCheck %s \
// RUN:   -check-prefix=OMP-HOST

#include <stdarg.h>

typedef struct { long double x; } ldbl128_s;

void foo_ld(long double);
void foo_fq(__float128);
void foo_ls(ldbl128_s);

// Verify cases when OpenMP target's and host's long-double semantics differ.

// OMP-TARGET-LABEL: define internal void @.omp_outlined.(
// OMP-TARGET: %[[CUR:[0-9a-zA-Z_.]+]] = load i8*, i8**
// OMP-TARGET: %[[V2:[0-9a-zA-Z_.]+]] = bitcast i8* %[[CUR]] to ppc_fp128*
// OMP-TARGET: %[[V3:[0-9a-zA-Z_.]+]] = load ppc_fp128, ppc_fp128* %[[V2]], align 8
// OMP-TARGET: call void @foo_ld(ppc_fp128 noundef %[[V3]])

// OMP-HOST-LABEL: define{{.*}} void @omp(
// OMP-HOST: %[[AP1:[0-9a-zA-Z_.]+]] = bitcast i8** %[[AP:[0-9a-zA-Z_.]+]] to i8*
// OMP-HOST: call void @llvm.va_start(i8* %[[AP1]])
// OMP-HOST: %[[CUR:[0-9a-zA-Z_.]+]] = load i8*, i8** %[[AP]], align 8
// OMP-HOST: %[[V0:[0-9a-zA-Z_.]+]] = ptrtoint i8* %[[CUR]] to i64
// OMP-HOST: %[[V1:[0-9a-zA-Z_.]+]] = add i64 %[[V0]], 15
// OMP-HOST: %[[V2:[0-9a-zA-Z_.]+]] = and i64 %[[V1]], -16
// OMP-HOST: %[[ALIGN:[0-9a-zA-Z_.]+]] = inttoptr i64 %[[V2]] to i8*
// OMP-HOST: %[[V3:[0-9a-zA-Z_.]+]] = bitcast i8* %[[ALIGN]] to fp128*
// OMP-HOST: %[[V4:[0-9a-zA-Z_.]+]] = load fp128, fp128* %[[V3]], align 16
// OMP-HOST: call void @foo_ld(fp128 noundef %[[V4]])
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

// IEEE-LABEL: define{{.*}} void @f128
// IEEE: %[[AP1:[0-9a-zA-Z_.]+]] = bitcast i8** %[[AP:[0-9a-zA-Z_.]+]] to i8*
// IEEE: call void @llvm.va_start(i8* %[[AP1]])
// IEEE: %[[CUR:[0-9a-zA-Z_.]+]] = load i8*, i8** %[[AP]]
// IEEE: %[[V0:[0-9a-zA-Z_.]+]] = ptrtoint i8* %[[CUR]] to i64
// IEEE: %[[V1:[0-9a-zA-Z_.]+]] = add i64 %[[V0]], 15
// IEEE: %[[V2:[0-9a-zA-Z_.]+]] = and i64 %[[V1]], -16
// IEEE: %[[ALIGN:[0-9a-zA-Z_.]+]] = inttoptr i64 %[[V2]] to i8*
// IEEE: %[[V3:[0-9a-zA-Z_.]+]] = bitcast i8* %[[ALIGN]] to fp128*
// IEEE: %[[V4:[0-9a-zA-Z_.]+]] = load fp128, fp128* %[[V3]], align 16
// IEEE: call void @foo_fq(fp128 noundef %[[V4]])
// IEEE: %[[AP2:[0-9a-zA-Z_.]+]] = bitcast i8** %[[AP]] to i8*
// IEEE: call void @llvm.va_end(i8* %[[AP2]])
void f128(int n, ...) {
  va_list ap;
  va_start(ap, n);
  foo_fq(va_arg(ap, __float128));
  va_end(ap);
}

// IEEE-LABEL: define{{.*}} void @long_double
// IEEE: %[[AP1:[0-9a-zA-Z_.]+]] = bitcast i8** %[[AP:[0-9a-zA-Z_.]+]] to i8*
// IEEE: call void @llvm.va_start(i8* %[[AP1]])
// IEEE: %[[CUR:[0-9a-zA-Z_.]+]] = load i8*, i8** %[[AP]]
// IEEE: %[[V0:[0-9a-zA-Z_.]+]] = ptrtoint i8* %[[CUR]] to i64
// IEEE: %[[V1:[0-9a-zA-Z_.]+]] = add i64 %[[V0]], 15
// IEEE: %[[V2:[0-9a-zA-Z_.]+]] = and i64 %[[V1]], -16
// IEEE: %[[ALIGN:[0-9a-zA-Z_.]+]] = inttoptr i64 %[[V2]] to i8*
// IEEE: %[[V3:[0-9a-zA-Z_.]+]] = bitcast i8* %[[ALIGN]] to fp128*
// IEEE: %[[V4:[0-9a-zA-Z_.]+]] = load fp128, fp128* %[[V3]], align 16
// IEEE: call void @foo_ld(fp128 noundef %[[V4]])
// IEEE: %[[AP2:[0-9a-zA-Z_.]+]] = bitcast i8** %[[AP]] to i8*
// IEEE: call void @llvm.va_end(i8* %[[AP2]])

// IBM-LABEL: define{{.*}} void @long_double
// IBM: %[[AP1:[0-9a-zA-Z_.]+]] = bitcast i8** %[[AP:[0-9a-zA-Z_.]+]] to i8*
// IBM: call void @llvm.va_start(i8*  %[[AP1]])
// IBM: %[[CUR:[0-9a-zA-Z_.]+]] = load i8*, i8** %[[AP]]
// IBM: %[[V3:[0-9a-zA-Z_.]+]] = bitcast i8* %[[CUR]] to ppc_fp128*
// IBM: %[[V4:[0-9a-zA-Z_.]+]] = load ppc_fp128, ppc_fp128* %[[V3]], align 8
// IBM: call void @foo_ld(ppc_fp128 noundef %[[V4]])
// IBM: %[[AP2:[0-9a-zA-Z_.]+]] = bitcast i8** %[[AP]] to i8*
// IBM: call void @llvm.va_end(i8* %[[AP2]])
void long_double(int n, ...) {
  va_list ap;
  va_start(ap, n);
  foo_ld(va_arg(ap, long double));
  va_end(ap);
}

// IEEE-LABEL: define{{.*}} void @long_double_struct
// IEEE: %[[AP1:[0-9a-zA-Z_.]+]] = bitcast i8** %[[AP:[0-9a-zA-Z_.]+]] to i8*
// IEEE: call void @llvm.va_start(i8* %[[AP1]])
// IEEE: %[[CUR:[0-9a-zA-Z_.]+]] = load i8*, i8** %[[AP]]
// IEEE: %[[P0:[0-9a-zA-Z_.]+]] = ptrtoint i8* %[[CUR]] to i64
// IEEE: %[[P1:[0-9a-zA-Z_.]+]] = add i64 %[[P0]], 15
// IEEE: %[[P2:[0-9a-zA-Z_.]+]] = and i64 %[[P1]], -16
// IEEE: %[[ALIGN:[0-9a-zA-Z_.]+]] = inttoptr i64 %[[P2]] to i8*
// IEEE: %[[V0:[0-9a-zA-Z_.]+]] = getelementptr inbounds i8, i8* %[[ALIGN]], i64 16
// IEEE: store i8* %[[V0]], i8** %[[AP]], align 8
// IEEE: %[[V1:[0-9a-zA-Z_.]+]] = bitcast i8* %[[ALIGN]] to %struct.ldbl128_s*
// IEEE: %[[V2:[0-9a-zA-Z_.]+]] = bitcast %struct.ldbl128_s* %[[TMP:[0-9a-zA-Z_.]+]] to i8*
// IEEE: %[[V3:[0-9a-zA-Z_.]+]] = bitcast %struct.ldbl128_s* %[[V1]] to i8*
// IEEE: call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %[[V2]], i8* align 16 %[[V3]], i64 16, i1 false)
// IEEE: %[[COERCE:[0-9a-zA-Z_.]+]] = getelementptr inbounds %struct.ldbl128_s, %struct.ldbl128_s* %[[TMP]], i32 0, i32 0
// IEEE: %[[V4:[0-9a-zA-Z_.]+]] = load fp128, fp128* %[[COERCE]], align 16
// IEEE: call void @foo_ls(fp128 inreg %[[V4]])
// IEEE: %[[AP2:[0-9a-zA-Z_.]+]] = bitcast i8** %[[AP]] to i8*
// IEEE: call void @llvm.va_end(i8* %[[AP2]])
void long_double_struct(int n, ...) {
  va_list ap;
  va_start(ap, n);
  foo_ls(va_arg(ap, ldbl128_s));
  va_end(ap);
}

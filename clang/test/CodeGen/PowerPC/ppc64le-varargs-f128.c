// RUN: %clang_cc1 -triple powerpc64le-unknown-linux-gnu -emit-llvm \
// RUN:   -target-cpu pwr9 -target-feature +float128 -mabi=ieeelongdouble \
// RUN:   -o - %s | FileCheck %s -check-prefix=IEEE
// RUN: %clang_cc1 -triple powerpc64le-unknown-linux-gnu -emit-llvm \
// RUN:   -target-cpu pwr9 -target-feature +float128 \
// RUN:   -o - %s | FileCheck %s -check-prefix=IBM

// RUN: %clang_cc1 -triple ppc64le -emit-llvm-bc %s -target-cpu pwr9 \
// RUN:   -target-feature +float128 -mabi=ieeelongdouble -fopenmp \
// RUN:   -fopenmp-targets=ppc64le -o %t-ppc-host.bc
// RUN: %clang_cc1 -triple ppc64le -aux-triple ppc64le %s -target-cpu pwr9 \
// RUN:   -target-feature +float128 -fopenmp -fopenmp-is-device -emit-llvm \
// RUN:   -fopenmp-host-ir-file-path %t-ppc-host.bc -o - | FileCheck %s \
// RUN:   -check-prefix=OMP-TARGET
// RUN: %clang_cc1 -triple ppc64le %t-ppc-host.bc -emit-llvm -o - | FileCheck %s \
// RUN:   -check-prefix=OMP-HOST

#include <stdarg.h>

typedef struct { long double x; } ldbl128_s;

void foo_ld(long double);
void foo_fq(__float128);
void foo_ls(ldbl128_s);

// Verify cases when OpenMP target's and host's long-double semantics differ.

// OMP-TARGET-LABEL: define internal void @.omp_outlined.(
// OMP-TARGET: %[[CUR:[0-9a-zA-Z_.]+]] = load ptr, ptr
// OMP-TARGET: %[[V3:[0-9a-zA-Z_.]+]] = load ppc_fp128, ptr %[[CUR]], align 8
// OMP-TARGET: call void @foo_ld(ppc_fp128 noundef %[[V3]])

// OMP-HOST-LABEL: define{{.*}} void @omp(
// OMP-HOST: call void @llvm.va_start(ptr %[[AP:[0-9a-zA-Z_.]+]])
// OMP-HOST: %[[CUR:[0-9a-zA-Z_.]+]] = load ptr, ptr %[[AP]], align 8
// OMP-HOST: %[[TMP0:[^ ]+]] = getelementptr inbounds i8, ptr %[[CUR]], i32 15
// OMP-HOST: %[[ALIGN:[^ ]+]] = call ptr @llvm.ptrmask.p0.i64(ptr %[[TMP0]], i64 -16)
// OMP-HOST: %[[V4:[0-9a-zA-Z_.]+]] = load fp128, ptr %[[ALIGN]], align 16
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
// IEEE: call void @llvm.va_start(ptr %[[AP:[0-9a-zA-Z_.]+]])
// IEEE: %[[CUR:[0-9a-zA-Z_.]+]] = load ptr, ptr %[[AP]]
// IEEE: %[[TMP0:[^ ]+]] = getelementptr inbounds i8, ptr %[[CUR]], i32 15
// IEEE: %[[ALIGN:[^ ]+]] = call ptr @llvm.ptrmask.p0.i64(ptr %[[TMP0]], i64 -16)
// IEEE: %[[V4:[0-9a-zA-Z_.]+]] = load fp128, ptr %[[ALIGN]], align 16
// IEEE: call void @foo_fq(fp128 noundef %[[V4]])
// IEEE: call void @llvm.va_end(ptr %[[AP]])
void f128(int n, ...) {
  va_list ap;
  va_start(ap, n);
  foo_fq(va_arg(ap, __float128));
  va_end(ap);
}

// IEEE-LABEL: define{{.*}} void @long_double
// IEEE: call void @llvm.va_start(ptr %[[AP:[0-9a-zA-Z_.]+]])
// IEEE: %[[CUR:[0-9a-zA-Z_.]+]] = load ptr, ptr %[[AP]]
// IEEE: %[[TMP0:[^ ]+]] = getelementptr inbounds i8, ptr %[[CUR]], i32 15
// IEEE: %[[ALIGN:[^ ]+]] = call ptr @llvm.ptrmask.p0.i64(ptr %[[TMP0]], i64 -16)
// IEEE: %[[V4:[0-9a-zA-Z_.]+]] = load fp128, ptr %[[ALIGN]], align 16
// IEEE: call void @foo_ld(fp128 noundef %[[V4]])
// IEEE: call void @llvm.va_end(ptr %[[AP]])

// IBM-LABEL: define{{.*}} void @long_double
// IBM: call void @llvm.va_start(ptr  %[[AP:[0-9a-zA-Z_.]+]])
// IBM: %[[CUR:[0-9a-zA-Z_.]+]] = load ptr, ptr %[[AP]]
// IBM: %[[V4:[0-9a-zA-Z_.]+]] = load ppc_fp128, ptr %[[CUR]], align 8
// IBM: call void @foo_ld(ppc_fp128 noundef %[[V4]])
// IBM: call void @llvm.va_end(ptr %[[AP]])
void long_double(int n, ...) {
  va_list ap;
  va_start(ap, n);
  foo_ld(va_arg(ap, long double));
  va_end(ap);
}

// IEEE-LABEL: define{{.*}} void @long_double_struct
// IEEE: call void @llvm.va_start(ptr %[[AP:[0-9a-zA-Z_.]+]])
// IEEE: %[[CUR:[0-9a-zA-Z_.]+]] = load ptr, ptr %[[AP]]
// IEEE: %[[TMP0:[^ ]+]] = getelementptr inbounds i8, ptr %[[CUR]], i32 15
// IEEE: %[[ALIGN:[^ ]+]] = call ptr @llvm.ptrmask.p0.i64(ptr %[[TMP0]], i64 -16)
// IEEE: %[[V0:[0-9a-zA-Z_.]+]] = getelementptr inbounds i8, ptr %[[ALIGN]], i64 16
// IEEE: store ptr %[[V0]], ptr %[[AP]], align 8
// IEEE: call void @llvm.memcpy.p0.p0.i64(ptr align 16 %[[TMP:[0-9a-zA-Z_.]+]], ptr align 16 %[[ALIGN]], i64 16, i1 false)
// IEEE: %[[COERCE:[0-9a-zA-Z_.]+]] = getelementptr inbounds %struct.ldbl128_s, ptr %[[TMP]], i32 0, i32 0
// IEEE: %[[V4:[0-9a-zA-Z_.]+]] = load fp128, ptr %[[COERCE]], align 16
// IEEE: call void @foo_ls(fp128 inreg %[[V4]])
// IEEE: call void @llvm.va_end(ptr %[[AP]])
void long_double_struct(int n, ...) {
  va_list ap;
  va_start(ap, n);
  foo_ls(va_arg(ap, ldbl128_s));
  va_end(ap);
}

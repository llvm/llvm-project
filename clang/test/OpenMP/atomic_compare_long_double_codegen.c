// RUN: %clang_cc1 -verify -triple x86_64-unknown-linux-gnu -fopenmp -x c -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -verify -triple x86_64-unknown-linux-gnu -fopenmp-simd -x c -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// expected-no-diagnostics

// GH140080: x86_fp80 occupies 16 bytes, so the cmpxchg must be on i128 with
// align 16, not on i80 (store size 10 is not a power of two).

// CHECK-LABEL: define {{.*}}void @f(
// CHECK:         [[LD_ADDR:%.*]] = alloca x86_fp80, align 16
// CHECK:         cmpxchg ptr [[LD_ADDR]], i128 0, i128 302222231531620438900736 monotonic monotonic, align 16
// CHECK-NOT:     i80
// CHECK:         ret void
void f(long double ld) {
#pragma omp atomic compare
  ld = ld == 0.0L ? 1.0L : ld;
}

// CHECK-LABEL: define {{.*}}void @g(
// CHECK:         [[X:%.*]] = load ptr, ptr %x.addr, align 8
// CHECK-NEXT:    [[E:%.*]] = load x86_fp80, ptr %e.addr, align 16
// CHECK-NEXT:    [[D:%.*]] = load x86_fp80, ptr %d.addr, align 16
// CHECK-NEXT:    [[E_BITS:%.*]] = bitcast x86_fp80 [[E]] to i80
// CHECK-NEXT:    [[E_INT:%.*]] = zext i80 [[E_BITS]] to i128
// CHECK-NEXT:    [[D_BITS:%.*]] = bitcast x86_fp80 [[D]] to i80
// CHECK-NEXT:    [[D_INT:%.*]] = zext i80 [[D_BITS]] to i128
// CHECK-NEXT:    cmpxchg ptr [[X]], i128 [[E_INT]], i128 [[D_INT]] monotonic monotonic, align 16
void g(long double *x, long double e, long double d) {
#pragma omp atomic compare
  *x = *x == e ? d : *x;
}

// CHECK-LABEL: define {{.*}}void @h(
// CHECK:         [[RES:%.*]] = cmpxchg ptr {{%.*}}, i128 {{%.*}}, i128 {{%.*}} monotonic monotonic, align 16
// CHECK-NEXT:    [[OLD_INT:%.*]] = extractvalue { i128, i1 } [[RES]], 0
// CHECK-NEXT:    [[OLD_BITS:%.*]] = trunc i128 [[OLD_INT]] to i80
// CHECK-NEXT:    [[OLD:%.*]] = bitcast i80 [[OLD_BITS]] to x86_fp80
// CHECK-NEXT:    store x86_fp80 [[OLD]], ptr {{%.*}}, align 16
void h(long double *x, long double *v, long double e, long double d) {
#pragma omp atomic compare capture
  {
    *v = *x;
    *x = *x == e ? d : *x;
  }
}

// CHECK-LABEL: define {{.*}}void @f_double(
// CHECK:         [[D_ADDR:%.*]] = alloca double, align 8
// CHECK:         cmpxchg ptr [[D_ADDR]], i64 0, i64 4607182418800017408 monotonic monotonic, align 8
void f_double(double d) {
#pragma omp atomic compare
  d = d == 0.0 ? 1.0 : d;
}

// SIMD-ONLY0-NOT: {{__kmpc|__tgt}}

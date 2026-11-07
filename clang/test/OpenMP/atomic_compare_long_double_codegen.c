// RUN: %clang_cc1 -verify -triple x86_64-unknown-linux-gnu -fopenmp -x c -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -verify -triple x86_64-unknown-linux-gnu -fopenmp -fopenmp-version=52 -x c -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -verify -triple x86_64-unknown-linux-gnu -fopenmp-simd -x c -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// expected-no-diagnostics

// GH140080: x86_fp80 holds 80 value bits in 16 bytes, so only the value bits
// are compared and the loaded pattern is the expected value of the cmpxchg.

// CHECK-LABEL: define {{.*}}void @f(
// CHECK:         [[LD_ADDR:%.*]] = alloca x86_fp80, align 16
// CHECK:         [[LOAD:%.*]] = load atomic i128, ptr [[LD_ADDR]] monotonic, align 16
// CHECK-NEXT:    br label %[[CONT:.*]]
// CHECK:       [[CONT]]:
// CHECK-NEXT:    [[CURR:%.*]] = phi i128 [ [[LOAD]], %entry ], [ [[CAS_OLD:%[0-9]+]], %[[CAS:[^ ]+]] ]
// CHECK-NEXT:    [[BITS:%.*]] = trunc i128 [[CURR]] to i80
// CHECK-NEXT:    [[EQ:%.*]] = icmp eq i80 [[BITS]], 0
// CHECK-NEXT:    br i1 [[EQ]], label %[[CAS]], label %[[EXIT:.*]]
// CHECK:       [[CAS]]:
// CHECK-NEXT:    [[RES:%.*]] = cmpxchg ptr [[LD_ADDR]], i128 [[CURR]], i128 302222231531620438900736 monotonic monotonic, align 16
// CHECK-NEXT:    [[CAS_OLD]] = extractvalue { i128, i1 } [[RES]], 0
// CHECK-NEXT:    [[OK:%.*]] = extractvalue { i128, i1 } [[RES]], 1
// CHECK-NEXT:    br i1 [[OK]], label %[[EXIT]], label %[[CONT]]
// CHECK:       [[EXIT]]:
// CHECK-NEXT:    [[OLD_INT:%.*]] = phi i128 [ [[CURR]], %[[CONT]] ], [ [[CAS_OLD]], %[[CAS]] ]
// CHECK-NEXT:    phi i1 [ false, %[[CONT]] ], [ true, %[[CAS]] ]
// CHECK-NEXT:    [[OLD_BITS:%.*]] = trunc i128 [[OLD_INT]] to i80
// CHECK-NEXT:    bitcast i80 [[OLD_BITS]] to x86_fp80
// CHECK-NEXT:    ret void
void f(long double ld) {
#pragma omp atomic compare
  ld = ld == 0.0L ? 1.0L : ld;
}

// CHECK-LABEL: define {{.*}}void @g(
// CHECK:         [[X:%.*]] = load ptr, ptr %x.addr, align 8
// CHECK-NEXT:    [[E:%.*]] = load x86_fp80, ptr %e.addr, align 16
// CHECK-NEXT:    [[D:%.*]] = load x86_fp80, ptr %d.addr, align 16
// CHECK-NEXT:    [[E_BITS:%.*]] = bitcast x86_fp80 [[E]] to i80
// CHECK-NEXT:    [[D_BITS:%.*]] = bitcast x86_fp80 [[D]] to i80
// CHECK-NEXT:    [[D_INT:%.*]] = zext i80 [[D_BITS]] to i128
// CHECK-NEXT:    [[LOAD:%.*]] = load atomic i128, ptr [[X]] monotonic, align 16
// CHECK:         [[CURR:%.*]] = phi i128 [ [[LOAD]], %entry ], [ [[CAS_OLD:%[0-9]+]], %{{.*}} ]
// CHECK-NEXT:    [[BITS:%.*]] = trunc i128 [[CURR]] to i80
// CHECK-NEXT:    icmp eq i80 [[BITS]], [[E_BITS]]
// CHECK:         [[RES:%.*]] = cmpxchg ptr [[X]], i128 [[CURR]], i128 [[D_INT]] monotonic monotonic, align 16
// CHECK-NEXT:    [[CAS_OLD]] = extractvalue { i128, i1 } [[RES]], 0
void g(long double *x, long double e, long double d) {
#pragma omp atomic compare
  *x = *x == e ? d : *x;
}

// CHECK-LABEL: define {{.*}}void @h(
// CHECK:         [[V:%.*]] = load ptr, ptr %v.addr, align 8
// CHECK:         cmpxchg ptr {{%.*}}, i128 {{%.*}}, i128 {{%.*}} monotonic monotonic, align 16
// CHECK:         [[OLD_INT:%.*]] = phi i128
// CHECK-NEXT:    phi i1
// CHECK-NEXT:    [[OLD_BITS:%.*]] = trunc i128 [[OLD_INT]] to i80
// CHECK-NEXT:    [[OLD:%.*]] = bitcast i80 [[OLD_BITS]] to x86_fp80
// CHECK-NEXT:    store x86_fp80 [[OLD]], ptr [[V]], align 16
void h(long double *x, long double *v, long double e, long double d) {
#pragma omp atomic compare capture
  {
    *v = *x;
    *x = *x == e ? d : *x;
  }
}

// v = x after the update: v gets d on success.
// CHECK-LABEL: define {{.*}}void @h_new(
// CHECK:         [[D:%.*]] = load x86_fp80, ptr %d.addr, align 16
// CHECK-NEXT:    [[V:%.*]] = load ptr, ptr %v.addr, align 8
// CHECK:         [[OK:%.*]] = phi i1 [ false,
// CHECK-NEXT:    [[OLD_BITS:%.*]] = trunc i128 {{%.*}} to i80
// CHECK-NEXT:    [[OLD:%.*]] = bitcast i80 [[OLD_BITS]] to x86_fp80
// CHECK-NEXT:    [[NEW:%.*]] = select i1 [[OK]], x86_fp80 [[D]], x86_fp80 [[OLD]]
// CHECK-NEXT:    store x86_fp80 [[NEW]], ptr [[V]], align 16
void h_new(long double *x, long double *v, long double e, long double d) {
#pragma omp atomic compare capture
  {
    if (*x == e) {
      *x = d;
    }
    *v = *x;
  }
}

// A signed r still receives 1, not -1.
// CHECK-LABEL: define {{.*}}void @r_signed(
// CHECK:         [[OK:%.*]] = phi i1 [ false,
// CHECK:         [[R:%.*]] = zext i1 [[OK]] to i32
// CHECK-NEXT:    store i32 [[R]], ptr %r, align 4
void r_signed(long double *x, long double e, long double d) {
  int r = 0;
#pragma omp atomic compare capture
  {
    r = *x == e;
    if (r) {
      *x = d;
    }
  }
}

struct __attribute__((packed)) P {
  char c;
  long double x;
};

// CHECK-LABEL: define {{.*}}void @p(
// CHECK:         [[X:%.*]] = getelementptr inbounds {{.*}}%struct.P, ptr {{%.*}}, i32 0, i32 1
// CHECK:         load atomic i128, ptr [[X]] monotonic, align 1
// CHECK:         cmpxchg ptr [[X]], i128 {{%.*}}, i128 {{%.*}} monotonic monotonic, align 1
void p(struct P *s, long double e, long double d) {
#pragma omp atomic compare
  s->x = s->x == e ? d : s->x;
}

// CHECK-LABEL: define {{.*}}void @f_double(
// CHECK:         [[D_ADDR:%.*]] = alloca double, align 8
// CHECK-NEXT:    store double
// CHECK-NEXT:    cmpxchg ptr [[D_ADDR]], i64 0, i64 4607182418800017408 monotonic monotonic, align 8
void f_double(double d) {
#pragma omp atomic compare
  d = d == 0.0 ? 1.0 : d;
}

// SIMD-ONLY0-NOT: {{__kmpc|__tgt}}

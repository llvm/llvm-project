// RUN: %clang_cc1 -triple x86_64-unknown-unknown -emit-llvm -o - %s | FileCheck %s

typedef double * __attribute__((align_value(64))) aligned_double;

void foo(aligned_double x, double * y __attribute__((align_value(32))),
         double & z __attribute__((align_value(128)))) { };
// CHECK: define void @_Z3fooPdS_Rd(double* align 64 %x, double* align 32 %y, double* align 128 dereferenceable(8) %z)

struct ad_struct {
  aligned_double a;
};

double *foo(ad_struct& x) {
// CHECK-LABEL: @_Z3fooR9ad_struct

// CHECK: [[PTRINT1:%.+]] = ptrtoint
// CHECK: [[MASKEDPTR1:%.+]] = and i64 [[PTRINT1]], 63
// CHECK: [[MASKCOND1:%.+]] = icmp eq i64 [[MASKEDPTR1]], 0
// CHECK: call void @llvm.assume(i1 [[MASKCOND1]])
  return x.a;
}

double *goo(ad_struct *x) {
// CHECK-LABEL: @_Z3gooP9ad_struct

// CHECK: [[PTRINT2:%.+]] = ptrtoint
// CHECK: [[MASKEDPTR2:%.+]] = and i64 [[PTRINT2]], 63
// CHECK: [[MASKCOND2:%.+]] = icmp eq i64 [[MASKEDPTR2]], 0
// CHECK: call void @llvm.assume(i1 [[MASKCOND2]])
  return x->a;
}

double *bar(aligned_double *x) {
// CHECK-LABEL: @_Z3barPPd

// CHECK: [[PTRINT3:%.+]] = ptrtoint
// CHECK: [[MASKEDPTR3:%.+]] = and i64 [[PTRINT3]], 63
// CHECK: [[MASKCOND3:%.+]] = icmp eq i64 [[MASKEDPTR3]], 0
// CHECK: call void @llvm.assume(i1 [[MASKCOND3]])
  return *x;
}

double *car(aligned_double &x) {
// CHECK-LABEL: @_Z3carRPd

// CHECK: [[PTRINT4:%.+]] = ptrtoint
// CHECK: [[MASKEDPTR4:%.+]] = and i64 [[PTRINT4]], 63
// CHECK: [[MASKCOND4:%.+]] = icmp eq i64 [[MASKEDPTR4]], 0
// CHECK: call void @llvm.assume(i1 [[MASKCOND4]])
  return x;
}

double *dar(aligned_double *x) {
// CHECK-LABEL: @_Z3darPPd

// CHECK: [[PTRINT5:%.+]] = ptrtoint
// CHECK: [[MASKEDPTR5:%.+]] = and i64 [[PTRINT5]], 63
// CHECK: [[MASKCOND5:%.+]] = icmp eq i64 [[MASKEDPTR5]], 0
// CHECK: call void @llvm.assume(i1 [[MASKCOND5]])
  return x[5];
}

aligned_double eep();
double *ret() {
// CHECK-LABEL: @_Z3retv

// CHECK: [[PTRINT6:%.+]] = ptrtoint
// CHECK: [[MASKEDPTR6:%.+]] = and i64 [[PTRINT6]], 63
// CHECK: [[MASKCOND6:%.+]] = icmp eq i64 [[MASKEDPTR6]], 0
// CHECK: call void @llvm.assume(i1 [[MASKCOND6]])
  return eep();
}

double **no1(aligned_double *x) {
// CHECK-LABEL: @_Z3no1PPd
  return x;
// CHECK-NOT: call void @llvm.assume
}

double *&no2(aligned_double &x) {
// CHECK-LABEL: @_Z3no2RPd
  return x;
// CHECK-NOT: call void @llvm.assume
}

double **no3(aligned_double &x) {
// CHECK-LABEL: @_Z3no3RPd
  return &x;
// CHECK-NOT: call void @llvm.assume
}

double no3(aligned_double x) {
// CHECK-LABEL: @_Z3no3Pd
  return *x;
// CHECK-NOT: call void @llvm.assume
}

double *no4(aligned_double x) {
// CHECK-LABEL: @_Z3no4Pd
  return x;
// CHECK-NOT: call void @llvm.assume
}


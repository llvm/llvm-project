// RUN: %clang_cc1 -triple i386-pc-elfiamcu -emit-llvm -o - %s | FileCheck %s

// CHECK: target datalayout = "e-m:e-p:32:32-p270:32:32-p271:32:32-p272:64:64-i64:32-f64:32-f128:32-n8:16:32-a:0:32-S32"
// CHECK: target triple = "i386-pc-elfiamcu"

void food(double *d);
void fooll(long long *ll);
void fooull(unsigned long long *ull);
void foold(long double *ld);

// CHECK-LABEL: define void @testdouble()
// CHECK: alloca double, align 4
void testdouble() {
  double d = 2.0;
  food(&d);
}

// CHECK-LABEL: define void @testlonglong()
// CHECK: alloca i64, align 4
void testlonglong() {
  long long ll = 2;
  fooll(&ll);
}

// CHECK-LABEL: define void @testunsignedlonglong()
// CHECK: alloca i64, align 4
void testunsignedlonglong() {
  unsigned long long ull = 2;
  fooull(&ull);	
}

// CHECK-LABEL: define void @testlongdouble()
// CHECK: alloca double, align 4
void testlongdouble() {
  long double ld = 2.0;
  foold(&ld);
}

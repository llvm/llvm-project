// RUN: %clang_cc1 -no-opaque-pointers -triple s390x-linux-gnu -O2 -emit-llvm \
// RUN:   -o - %s 2>&1 | FileCheck %s
// REQUIRES: systemz-registered-target

long *A;
long Idx;
unsigned long Addr;

unsigned long fun_BD12_p() {
// CHECK-LABEL: define{{.*}} i64 @fun_BD12_p()
// CHECK: call i64 asm "lay $0, $1", "=r,p"(i64* nonnull %arrayidx)
  asm("lay %0, %1" : "=r" (Addr) : "p" (&A[100]));
  return Addr;
}

unsigned long fun_BDX12_p() {
// CHECK-LABEL: define{{.*}} i64 @fun_BDX12_p()
// CHECK: call i64 asm "lay $0, $1", "=r,p"(i64* %arrayidx)
  asm("lay %0, %1" : "=r" (Addr) : "p" (&A[Idx + 100]));
  return Addr;
}

unsigned long fun_BD20_p() {
// CHECK-LABEL: define{{.*}} i64 @fun_BD20_p()
// CHECK: call i64 asm "lay $0, $1", "=r,p"(i64* nonnull %arrayidx)
  asm("lay %0, %1" : "=r" (Addr) : "p" (&A[1000]));
  return Addr;
}

unsigned long fun_BDX20_p() {
// CHECK-LABEL: define{{.*}} i64 @fun_BDX20_p()
// CHECK: call i64 asm "lay $0, $1", "=r,p"(i64* %arrayidx)
  asm("lay %0, %1" : "=r" (Addr) : "p" (&A[Idx + 1000]));
  return Addr;
}

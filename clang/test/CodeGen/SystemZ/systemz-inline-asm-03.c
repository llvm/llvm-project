// RUN: %clang_cc1 -no-opaque-pointers -triple s390x-linux-gnu -O2 -emit-llvm \
// RUN:   -o - %s 2>&1 | FileCheck %s
// REQUIRES: systemz-registered-target

long *A;
long Idx;
unsigned long Addr;

unsigned long fun_BD12_Q() {
// CHECK-LABEL: define{{.*}} i64 @fun_BD12_Q()
// CHECK: call i64 asm "lay $0, $1", "=r,^ZQ"(i64* nonnull %arrayidx)
  asm("lay %0, %1" : "=r" (Addr) : "ZQ" (&A[100]));
  return Addr;
}

unsigned long fun_BD12_R() {
// CHECK-LABEL: define{{.*}} i64 @fun_BD12_R()
// CHECK: call i64 asm "lay $0, $1", "=r,^ZR"(i64* nonnull %arrayidx)
  asm("lay %0, %1" : "=r" (Addr) : "ZR" (&A[100]));
  return Addr;
}

unsigned long fun_BD12_S() {
// CHECK-LABEL: define{{.*}} i64 @fun_BD12_S()
// CHECK: call i64 asm "lay $0, $1", "=r,^ZS"(i64* nonnull %arrayidx)
  asm("lay %0, %1" : "=r" (Addr) : "ZS" (&A[100]));
  return Addr;
}

unsigned long fun_BD12_T() {
// CHECK-LABEL: define{{.*}} i64 @fun_BD12_T()
// CHECK: call i64 asm "lay $0, $1", "=r,^ZT"(i64* nonnull %arrayidx)
  asm("lay %0, %1" : "=r" (Addr) : "ZT" (&A[100]));
  return Addr;
}

unsigned long fun_BD12_p() {
// CHECK-LABEL: define{{.*}} i64 @fun_BD12_p()
// CHECK: call i64 asm "lay $0, $1", "=r,p"(i64* nonnull %arrayidx)
  asm("lay %0, %1" : "=r" (Addr) : "p" (&A[100]));
  return Addr;
}

unsigned long fun_BDX12_Q() {
// CHECK-LABEL: define{{.*}} i64 @fun_BDX12_Q()
// CHECK: call i64 asm "lay $0, $1", "=r,^ZQ"(i64* %arrayidx)
  asm("lay %0, %1" : "=r" (Addr) : "ZQ" (&A[Idx + 100]));
  return Addr;
}

unsigned long fun_BDX12_R() {
// CHECK-LABEL: define{{.*}} i64 @fun_BDX12_R()
// CHECK: call i64 asm "lay $0, $1", "=r,^ZR"(i64* %arrayidx)
  asm("lay %0, %1" : "=r" (Addr) : "ZR" (&A[Idx + 100]));
  return Addr;
}

unsigned long fun_BDX12_S() {
// CHECK-LABEL: define{{.*}} i64 @fun_BDX12_S()
// CHECK: call i64 asm "lay $0, $1", "=r,^ZS"(i64* %arrayidx)
  asm("lay %0, %1" : "=r" (Addr) : "ZS" (&A[Idx + 100]));
  return Addr;
}

unsigned long fun_BDX12_T() {
// CHECK-LABEL: define{{.*}} i64 @fun_BDX12_T()
// CHECK: call i64 asm "lay $0, $1", "=r,^ZT"(i64* %arrayidx)
  asm("lay %0, %1" : "=r" (Addr) : "ZT" (&A[Idx + 100]));
  return Addr;
}

unsigned long fun_BDX12_p() {
// CHECK-LABEL: define{{.*}} i64 @fun_BDX12_p()
// CHECK: call i64 asm "lay $0, $1", "=r,p"(i64* %arrayidx)
  asm("lay %0, %1" : "=r" (Addr) : "p" (&A[Idx + 100]));
  return Addr;
}

unsigned long fun_BD20_Q() {
// CHECK-LABEL: define{{.*}} i64 @fun_BD20_Q()
// CHECK: call i64 asm "lay $0, $1", "=r,^ZQ"(i64* nonnull %arrayidx)
  asm("lay %0, %1" : "=r" (Addr) : "ZQ" (&A[1000]));
  return Addr;
}

unsigned long fun_BD20_R() {
// CHECK-LABEL: define{{.*}} i64 @fun_BD20_R()
// CHECK: call i64 asm "lay $0, $1", "=r,^ZR"(i64* nonnull %arrayidx)
  asm("lay %0, %1" : "=r" (Addr) : "ZR" (&A[1000]));
  return Addr;
}

unsigned long fun_BD20_S() {
// CHECK-LABEL: define{{.*}} i64 @fun_BD20_S()
// CHECK: call i64 asm "lay $0, $1", "=r,^ZS"(i64* nonnull %arrayidx)
  asm("lay %0, %1" : "=r" (Addr) : "ZS" (&A[1000]));
  return Addr;
}

unsigned long fun_BD20_T() {
// CHECK-LABEL: define{{.*}} i64 @fun_BD20_T()
// CHECK: call i64 asm "lay $0, $1", "=r,^ZT"(i64* nonnull %arrayidx)
  asm("lay %0, %1" : "=r" (Addr) : "ZT" (&A[1000]));
  return Addr;
}

unsigned long fun_BD20_p() {
// CHECK-LABEL: define{{.*}} i64 @fun_BD20_p()
// CHECK: call i64 asm "lay $0, $1", "=r,p"(i64* nonnull %arrayidx)
  asm("lay %0, %1" : "=r" (Addr) : "p" (&A[1000]));
  return Addr;
}

unsigned long fun_BDX20_Q() {
// CHECK-LABEL: define{{.*}} i64 @fun_BDX20_Q()
// CHECK: call i64 asm "lay $0, $1", "=r,^ZQ"(i64* %arrayidx)
  asm("lay %0, %1" : "=r" (Addr) : "ZQ" (&A[Idx + 1000]));
  return Addr;
}

unsigned long fun_BDX20_R() {
// CHECK-LABEL: define{{.*}} i64 @fun_BDX20_R()
// CHECK: call i64 asm "lay $0, $1", "=r,^ZR"(i64* %arrayidx)
  asm("lay %0, %1" : "=r" (Addr) : "ZR" (&A[Idx + 1000]));
  return Addr;
}

unsigned long fun_BDX20_S() {
// CHECK-LABEL: define{{.*}} i64 @fun_BDX20_S()
// CHECK: call i64 asm "lay $0, $1", "=r,^ZS"(i64* %arrayidx)
  asm("lay %0, %1" : "=r" (Addr) : "ZS" (&A[Idx + 1000]));
  return Addr;
}

unsigned long fun_BDX20_T() {
// CHECK-LABEL: define{{.*}} i64 @fun_BDX20_T()
// CHECK: call i64 asm "lay $0, $1", "=r,^ZT"(i64* %arrayidx)
  asm("lay %0, %1" : "=r" (Addr) : "ZT" (&A[Idx + 1000]));
  return Addr;
}

unsigned long fun_BDX20_p() {
// CHECK-LABEL: define{{.*}} i64 @fun_BDX20_p()
// CHECK: call i64 asm "lay $0, $1", "=r,p"(i64* %arrayidx)
  asm("lay %0, %1" : "=r" (Addr) : "p" (&A[Idx + 1000]));
  return Addr;
}

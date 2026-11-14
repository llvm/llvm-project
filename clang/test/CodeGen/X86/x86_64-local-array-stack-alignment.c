// RUN: %clang_cc1 -triple x86_64-linux-gnu -mstack-alignment=8 -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,ALIGN8
// RUN: %clang_cc1 -triple x86_64-linux-gnu -mstack-alignment=16 -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,ALIGN16
// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,ALIGN16

// Globals keep 16-byte ABI alignment even under -mstack-alignment=8.
// CHECK: @foo.sbuf = internal global [16 x i8] zeroinitializer, align 16
// CHECK: @gbuf ={{.*}} global [16 x i8] zeroinitializer, align 16
char gbuf[16];

void sink(void *);

// CHECK-LABEL: define{{.*}} void @foo(
void foo(int n) {
  // FIXME: Under -mstack-alignment=8, these local arrays should not be
  // promoted to 16-byte alignment (which forces dynamic stack realignment).
  // ALIGN8: %buf = alloca [16 x i8], align 16
  // ALIGN16: %buf = alloca [16 x i8], align 16
  char buf[16];

  // FIXME: Should be align 4 under -mstack-alignment=8.
  // ALIGN8: %ibuf = alloca [4 x i32], align 16
  // ALIGN16: %ibuf = alloca [4 x i32], align 16
  int ibuf[4];

  // ALIGN8: %abuf = alloca [16 x i8], align 16
  // ALIGN16: %abuf = alloca [16 x i8], align 16
  __attribute__((aligned(16))) char abuf[16];

  // FIXME: Should be align 1 under -mstack-alignment=8.
  // ALIGN8: %vla = alloca i8, i64 %{{.*}}, align 16
  // ALIGN16: %vla = alloca i8, i64 %{{.*}}, align 16
  char vla[n];

  static char sbuf[16];

  sink(buf);
  sink(ibuf);
  sink(abuf);
  sink(vla);
  sink(sbuf);
}

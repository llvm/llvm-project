// RUN: %clang_cc1 -S -emit-llvm -o - -O2 -disable-llvm-passes %s      | FileCheck %s --implicit-check-not="call void @llvm.lifetime" --check-prefixes=CHECK,O2
// RUN: %clang_cc1 -S -emit-llvm -o - -O2 -disable-lifetime-markers %s | FileCheck %s --implicit-check-not="call void @llvm.lifetime" --check-prefixes=CHECK
// RUN: %clang_cc1 -S -emit-llvm -o - -O0 %s                           | FileCheck %s --implicit-check-not="call void @llvm.lifetime" --check-prefixes=CHECK 

extern int bar(char *A, int n);

// CHECK-LABEL: @no_switch_bypass
extern "C" void no_switch_bypass(int n) {
  // O2: call void @llvm.lifetime.start.p0(i64 4,
  switch (n += 1; int b=n) {
  case 1: {
    // O2: call void @llvm.lifetime.start.p0(i64 1,
    // O2: call void @llvm.lifetime.end.p0(i64 1,
    char x;
    bar(&x, 1);
    break;
  }
  case 2:
    n = n;
    // O2: call void @llvm.lifetime.start.p0(i64 5,
    // O2: call void @llvm.lifetime.end.p0(i64 5,
    char y[5];
    bar(y, 5);
    break;
  }
  // O2: call void @llvm.lifetime.end.p0(i64 4,
}

// CHECK-LABEL: @switch_bypass
extern "C" void switch_bypass(int n) {
  // O2: call void @llvm.lifetime.start.p0(i64 4,
  // O2: call void @llvm.lifetime.end.p0(i64 4,
  switch (n += 1; int b=n) {
  case 1:
    n = n;
    char x;
    bar(&x, 1);
    break;
  case 2:
    bar(&x, 1);
    break;
  }
}

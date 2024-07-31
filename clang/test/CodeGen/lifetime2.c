// RUN: %clang_cc1 -emit-llvm -o - -O2 -disable-llvm-passes %s      | FileCheck %s --implicit-check-not="call void @llvm.lifetime" --check-prefixes=CHECK,O2
// RUN: %clang_cc1 -emit-llvm -o - -O2 -disable-lifetime-markers %s | FileCheck %s --implicit-check-not="call void @llvm.lifetime" --check-prefixes=CHECK
// RUN: %clang_cc1 -emit-llvm -o - -O0 %s                           | FileCheck %s --implicit-check-not="call void @llvm.lifetime" --check-prefixes=CHECK 

extern int bar(char *A, int n);

// CHECK-LABEL: @foo
int foo (int n) {
  if (n) {
// O2: call void @llvm.lifetime.start.p0(i64 100,
    char A[100];
    return bar(A, 1);
// O2: call void @llvm.lifetime.end.p0(i64 100,
  } else {
// O2: call void @llvm.lifetime.start.p0(i64 100,
    char A[100];
    return bar(A, 2);
// O2: call void @llvm.lifetime.end.p0(i64 100,
  }
}

// CHECK-LABEL: @no_goto_bypass
void no_goto_bypass(void) {
  // O2: call void @llvm.lifetime.start.p0(i64 1,
  char x;
l1:
  bar(&x, 1);
  char y[5];
  bar(y, 5);
  goto l1;
  // Infinite loop
}

// CHECK-LABEL: @goto_bypass
void goto_bypass(void) {
  {
    char x;
  l1:
    bar(&x, 1);
  }
  goto l1;
}

// CHECK-LABEL: @no_switch_bypass
void no_switch_bypass(int n) {
  switch (n) {
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
}

// CHECK-LABEL: @switch_bypass
void switch_bypass(int n) {
  switch (n) {
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

// CHECK-LABEL: @indirect_jump
void indirect_jump(int n) {
  char x;
  void *T[] = {&&L};
  goto *T[n];
L:
  bar(&x, 1);
}

extern void foo2(int p);

// O2-LABEL: @jump_backward_over_declaration(
int jump_backward_over_declaration(int a) {
  int *p = 0;
// O2: call void @llvm.lifetime.start.p0(
label1:
  if (p) {
    foo2(*p);
    return 0;
  }

  int i = 999;
  if (a != 2) {
    p = &i;
    goto label1;
  }
  return -1;
// O2: call void @llvm.lifetime.end.p0(
}

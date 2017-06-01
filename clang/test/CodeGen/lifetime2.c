// RUN: %clang_cc1 -S -emit-llvm -o - -O2 -disable-llvm-passes %s | FileCheck %s -check-prefixes=CHECK,O2
// RUN: %clang_cc1 -S -emit-llvm -o - -O2 -disable-lifetime-markers %s \
// RUN:       | FileCheck %s -check-prefixes=CHECK,O0
// RUN: %clang_cc1 -S -emit-llvm -o - -O0 %s | FileCheck %s -check-prefixes=CHECK,O0

extern int bar(char *A, int n);

// CHECK-LABEL: @foo
// O0-NOT: @llvm.lifetime.start
int foo (int n) {
  if (n) {
// O2: @llvm.lifetime.start
    char A[100];
    return bar(A, 1);
  } else {
// O2: @llvm.lifetime.start
    char A[100];
    return bar(A, 2);
  }
}

// CHECK-LABEL: @no_goto_bypass
void no_goto_bypass() {
  // O2: @llvm.lifetime.start(i64 1
  char x;
l1:
  bar(&x, 1);
  char y[5];
  bar(y, 5);
  goto l1;
  // Infinite loop
  // O2-NOT: @llvm.lifetime.end(
}

// CHECK-LABEL: @goto_bypass
void goto_bypass() {
  {
    // O2-NOT: @llvm.lifetime.start(i64 1
    // O2-NOT: @llvm.lifetime.end(i64 1
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
    // O2: @llvm.lifetime.start(i64 1
    // O2: @llvm.lifetime.end(i64 1
    char x;
    bar(&x, 1);
    break;
  }
  case 2:
    n = n;
    // O2: @llvm.lifetime.start(i64 5
    // O2: @llvm.lifetime.end(i64 5
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
    // O2-NOT: @llvm.lifetime.start(i64 1
    // O2-NOT: @llvm.lifetime.end(i64 1
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
  // O2-NOT: @llvm.lifetime
  void *T[] = {&&L};
  goto *T[n];
L:
  bar(&x, 1);
}

// O2-LABEL: @jump_backward_over_declaration(
// O2: %[[p:.*]] = alloca i32*
// O2: %[[v0:.*]] = bitcast i32** %[[p]] to i8*
// O2: call void @llvm.lifetime.start(i64 {{.*}}, i8* %[[v0]])
// O2-NOT: call void @llvm.lifetime.start(

extern void foo2(int p);

int jump_backward_over_declaration(int a) {
  int *p = 0;
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
}

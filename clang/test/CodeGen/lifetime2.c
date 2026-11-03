// RUN: %clang_cc1 -emit-llvm -o - -O2 -disable-llvm-passes %s      | FileCheck %s --implicit-check-not="call void @llvm.lifetime" --check-prefixes=CHECK,O2
// RUN: %clang_cc1 -emit-llvm -o - -O2 -disable-lifetime-markers %s | FileCheck %s --implicit-check-not="call void @llvm.lifetime" --check-prefixes=CHECK
// RUN: %clang_cc1 -emit-llvm -o - -O0 %s                           | FileCheck %s --implicit-check-not="call void @llvm.lifetime" --check-prefixes=CHECK 

extern int bar(char *A, int n);

// CHECK-LABEL: @foo
int foo (int n) {
  if (n) {
// O2: call void @llvm.lifetime.start.p0(
    char A[100];
    return bar(A, 1);
// O2: call void @llvm.lifetime.end.p0(
  } else {
// O2: call void @llvm.lifetime.start.p0(
    char A[100];
    return bar(A, 2);
// O2: call void @llvm.lifetime.end.p0(
  }
}

// CHECK-LABEL: @no_goto_bypass
void no_goto_bypass(void) {
  // O2: call void @llvm.lifetime.start.p0(
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
    // O2: call void @llvm.lifetime.start.p0(
    // O2: call void @llvm.lifetime.end.p0(
    char x;
    bar(&x, 1);
    break;
  }
  case 2:
    n = n;
    // O2: call void @llvm.lifetime.start.p0(
    // O2: call void @llvm.lifetime.end.p0(
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

// CHECK-LABEL: @asm_goto_bypass
void asm_goto_bypass(void) {
  __asm__ goto("" : : : : l1);
  char x;
l1:
  bar(&x, 1);
}

// CHECK-LABEL: @asm_goto_multi_label_bypass
void asm_goto_multi_label_bypass(void) {
  __asm__ goto("" : : : : l1, l2);
l1:
  return;
  {
    char x;
  l2:
    bar(&x, 1);
  }
}

// CHECK-LABEL: @asm_goto_backward_bypass
void asm_goto_backward_bypass(void) {
  {
    char x;
  l1:
    bar(&x, 1);
  }
  __asm__ goto("" : : : : l1);
}

// CHECK-LABEL: @no_asm_goto_bypass
void no_asm_goto_bypass(void) {
  __asm__("");
  // O2: call void @llvm.lifetime.start.p0(
  char x;
  bar(&x, 1);
  __asm__ goto("" : : : : l1);
l1:
  // O2: call void @llvm.lifetime.end.p0(
  return;
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

// CHECK-LABEL: @stmt_expr_label(
void stmt_expr_label(int n) {
  ({
    __label__ l;
  l:
    if (n--)
      goto l;
  });
  // O2: call void @llvm.lifetime.start.p0(
  char x;
  bar(&x, 1);
  // O2: call void @llvm.lifetime.end.p0(
}

// CHECK-LABEL: @label_without_backward_jump(
void label_without_backward_jump(int n) {
  if (n)
    goto l1;
  bar(0, 0);
l1:;
  // O2: call void @llvm.lifetime.start.p0(
  char x;
  bar(&x, 1);
  // O2: call void @llvm.lifetime.end.p0(
}

// CHECK-LABEL: @inner_block_backward_goto_to_outer(
void inner_block_backward_goto_to_outer(int n) {
l1:
  if (n <= 0)
    return;
  {
    // O2: call void @llvm.lifetime.start.p0(
    char x;
    bar(&x, 1);
    // O2: call void @llvm.lifetime.end.p0(
    --n;
    goto l1;
  }
}

// Verify bypass detection runs on non-FunctionDecl bodies (CapturedDecl):
// `p` before `label1` gets lifetime markers, while `i` crossed by `goto label1`
// does not.
// CHECK-LABEL: @captured_stmt_backward_goto(
// O2-LABEL: define internal void @__captured_stmt(
// O2: %p = alloca ptr
// O2: %i = alloca i32
// O2: call void @llvm.lifetime.start.p0(ptr %p)
// O2: call void @llvm.lifetime.end.p0(ptr %p)
void captured_stmt_backward_goto(void) {
  #pragma clang __debug captured
  {
    int *p = 0;
  label1:
    if (p) {
      foo2(*p);
      goto out;
    }
    int i = 999;
    p = &i;
    goto label1;
  out:;
  }
}

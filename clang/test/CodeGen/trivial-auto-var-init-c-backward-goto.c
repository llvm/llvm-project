// RUN: %clang_cc1 -triple x86_64-unknown-unknown -ftrivial-auto-var-init=zero %s -emit-llvm -o - | FileCheck %s --check-prefix=ZERO
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -ftrivial-auto-var-init=pattern %s -emit-llvm -o - | FileCheck %s --check-prefix=PATTERN

// In C, a bypassed variable's lifetime begins at entry into its enclosing
// block (C6.2.4p6), so it is initialized at that block's entry: once for a
// function-scoped variable, every iteration for a loop-scoped one. C++ differs
// (lifetime restarts on scope re-entry); see CodeGenCXX/trivial-auto-var-init.cpp.

void use_int(int *);

// Not bypassed: declaration is reached each iteration, so init is at BEGIN.
// ZERO-LABEL: define {{.*}}@backward_goto_pointer(
// ZERO: entry:
// ZERO-NOT: !annotation
// ZERO: BEGIN:
// ZERO: store ptr null, ptr %p, {{.*}}!annotation [[AUTO_INIT:!.+]]
// PATTERN-LABEL: define {{.*}}@backward_goto_pointer(
// PATTERN: entry:
// PATTERN-NOT: !annotation
// PATTERN: BEGIN:
// PATTERN: store ptr inttoptr (i64 -6148914691236517206 to ptr), ptr %p, {{.*}}!annotation [[AUTO_INIT:!.+]]
int backward_goto_pointer(void) {
  int b = 0;
BEGIN:;
  int *p;
  if (b)
    *p = 10;
  p = &b;
  if (!b) {
    b = 1;
    goto BEGIN;
  }
  return b;
}

// Scalar variant of the above.
// ZERO-LABEL: define {{.*}}@backward_goto_scalar(
// ZERO: entry:
// ZERO-NOT: !annotation
// ZERO: BEGIN:
// ZERO: store i32 0, ptr %c, {{.*}}!annotation [[AUTO_INIT]]
// PATTERN-LABEL: define {{.*}}@backward_goto_scalar(
// PATTERN: entry:
// PATTERN-NOT: !annotation
// PATTERN: BEGIN:
// PATTERN: store i32 -1431655766, ptr %c, {{.*}}!annotation [[AUTO_INIT]]
int backward_goto_scalar(void) {
  int b = 0;
BEGIN:;
  int c;
  if (b) return c;
  c = 5;
  b = 1;
  goto BEGIN;
}

// Bypassed, function-scoped: init once in entry, so p = &b survives the
// backward goto (returns 10).
// ZERO-LABEL: define {{.*}}@backward_goto_around_decl(
// ZERO: entry:
// ZERO: store ptr null, ptr %p, {{.*}}!annotation [[AUTO_INIT]]
// ZERO: BEGIN:
// ZERO-NOT: !annotation
// ZERO: ret
// PATTERN-LABEL: define {{.*}}@backward_goto_around_decl(
// PATTERN: entry:
// PATTERN: store ptr inttoptr (i64 -6148914691236517206 to ptr), ptr %p, {{.*}}!annotation [[AUTO_INIT]]
// PATTERN: BEGIN:
// PATTERN-NOT: !annotation
// PATTERN: ret
int backward_goto_around_decl(void) {
  int b = 0;
BEGIN:;
  goto CONT;
  int *p;
CONT:
  if (b)
    *p = 10;
  p = &b;
  if (!b) {
    b = 1;
    goto BEGIN;
  }
  return b;
}

// Loop-scoped: init at the loop body's entry (while.body), so it reruns each
// iteration rather than once in entry.
// ZERO-LABEL: define {{.*}}@loop_bypass(
// ZERO: entry:
// ZERO-NOT: !annotation
// ZERO: while.body:
// ZERO: store i32 0, ptr %x, {{.*}}!annotation [[AUTO_INIT]]
// PATTERN-LABEL: define {{.*}}@loop_bypass(
// PATTERN: entry:
// PATTERN-NOT: !annotation
// PATTERN: while.body:
// PATTERN: store i32 -1431655766, ptr %x, {{.*}}!annotation [[AUTO_INIT]]
void loop_bypass(void) {
  while (1) {
    goto X;
    int x;
  X:
    use_int(&x);
  }
}

// Switch bypass, function-scoped: init once in entry before the dispatch.
// ZERO-LABEL: define {{.*}}@switch_bypass(
// ZERO: entry:
// ZERO: store i32 0, ptr %x, {{.*}}!annotation [[AUTO_INIT]]
// ZERO: switch i32
// PATTERN-LABEL: define {{.*}}@switch_bypass(
// PATTERN: entry:
// PATTERN: store i32 -1431655766, ptr %x, {{.*}}!annotation [[AUTO_INIT]]
// PATTERN: switch i32
int switch_bypass(int c) {
  switch (c) {
    int x;
  case 0:
    x = 1;
    use_int(&x);
    return x;
  default:
    use_int(&x);
    return x;
  }
}

// Switch bypass in a loop: init at the dispatch block (while.body), per
// iteration.
// ZERO-LABEL: define {{.*}}@switch_bypass_in_loop(
// ZERO: entry:
// ZERO-NOT: !annotation
// ZERO: while.body:
// ZERO: store i32 0, ptr %x, {{.*}}!annotation [[AUTO_INIT]]
// ZERO: switch i32
// PATTERN-LABEL: define {{.*}}@switch_bypass_in_loop(
// PATTERN: entry:
// PATTERN-NOT: !annotation
// PATTERN: while.body:
// PATTERN: store i32 -1431655766, ptr %x, {{.*}}!annotation [[AUTO_INIT]]
// PATTERN: switch i32
void switch_bypass_in_loop(int c) {
  while (1) {
    switch (c) {
      int x;
    case 0:
      x = 1;
      use_int(&x);
      break;
    default:
      use_int(&x);
      break;
    }
  }
}

// Computed goto: sources unknown, so init once in entry.
// ZERO-LABEL: define {{.*}}@computed_goto(
// ZERO: entry:
// ZERO: store i32 0, ptr %y, {{.*}}!annotation [[AUTO_INIT]]
// ZERO: indirectbr
// PATTERN-LABEL: define {{.*}}@computed_goto(
// PATTERN: entry:
// PATTERN: store i32 -1431655766, ptr %y, {{.*}}!annotation [[AUTO_INIT]]
// PATTERN: indirectbr
void computed_goto(int n) {
  void *t[] = {&&L1, &&L2};
  goto *t[n];
  int y;
L1:
  use_int(&y);
  return;
L2:
  return;
}

// Nested loops: init at the inner body's entry (while.body3), per inner
// iteration -- not in entry or the outer body.
// ZERO-LABEL: define {{.*}}@nested_loops(
// ZERO: entry:
// ZERO-NOT: store {{.*}}%x{{.*}}!annotation
// ZERO: while.body3:
// ZERO: store i32 0, ptr %x, {{.*}}!annotation [[AUTO_INIT]]
// PATTERN-LABEL: define {{.*}}@nested_loops(
// PATTERN: entry:
// PATTERN-NOT: store {{.*}}%x{{.*}}!annotation
// PATTERN: while.body3:
// PATTERN: store i32 -1431655766, ptr %x, {{.*}}!annotation [[AUTO_INIT]]
void nested_loops(int n) {
  while (n) {
    while (n) {
      goto X;
      int x;
    X:
      use_int(&x);
      n--;
    }
  }
}

// Nested loops + switch: init at the inner dispatch block (while.body3), per
// inner iteration.
// ZERO-LABEL: define {{.*}}@nested_loops_switch(
// ZERO: entry:
// ZERO-NOT: store {{.*}}%x{{.*}}!annotation
// ZERO: while.body3:
// ZERO: store i32 0, ptr %x, {{.*}}!annotation [[AUTO_INIT]]
// ZERO: switch i32
// PATTERN-LABEL: define {{.*}}@nested_loops_switch(
// PATTERN: entry:
// PATTERN-NOT: store {{.*}}%x{{.*}}!annotation
// PATTERN: while.body3:
// PATTERN: store i32 -1431655766, ptr %x, {{.*}}!annotation [[AUTO_INIT]]
// PATTERN: switch i32
void nested_loops_switch(int n, int c) {
  while (n) {
    while (n) {
      switch (c) {
        int x;
      case 0:
        x = 1;
        use_int(&x);
        break;
      default:
        use_int(&x);
        break;
      }
      n--;
    }
  }
}

// Computed goto with multiple scopes: sources unknown, so every bypassed
// variable gets a single function-scope init in entry. The switch case targets
// must not add a second init store for %x.
// ZERO-LABEL: define {{.*}}@computed_goto_multi_scope(
// ZERO: entry:
// ZERO: store i32 0, ptr %x, {{.*}}!annotation [[AUTO_INIT]]
// ZERO: indirectbr
// ZERO-NOT: store i32 0, ptr %x, {{.*}}!annotation
// PATTERN-LABEL: define {{.*}}@computed_goto_multi_scope(
// PATTERN: entry:
// PATTERN: store i32 -1431655766, ptr %x, {{.*}}!annotation [[AUTO_INIT]]
// PATTERN: indirectbr
// PATTERN-NOT: store i32 -1431655766, ptr %x, {{.*}}!annotation
void computed_goto_multi_scope(int n, int c) {
  void *t[] = {&&L1, &&L2};
  goto *t[n];
  int x;
  switch (c) {
  case 0:
  L1:
    use_int(&x);
    break;
  default:
  L2:
    use_int(&x);
    break;
  }
}

// ZERO: [[AUTO_INIT]] = !{!"auto-init"}
// PATTERN: [[AUTO_INIT]] = !{!"auto-init"}

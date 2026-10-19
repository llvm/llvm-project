// RUN: %clang_cc1 -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -x c++ -emit-llvm -o - %s | FileCheck %s

#ifdef __cplusplus
extern "C" {
#endif

// Break in the increment exits the outer loop.
// CHECK-LABEL: define {{.*}}void @pr8880_cg_1(
// CHECK:       for.inc:
// CHECK:         br label %for.end6
// CHECK:       for.end:
// CHECK:       for.inc4:
// CHECK:       for.end6:
void pr8880_cg_1(int *iptr) {
  int i, j;
  for (i = 2; i != 10 ; i++ )
    for (j = 3 ; j < 22; (void)({ ++j; break; j;})) {
      *iptr = 7;
    }
}

// Continue in the increment targets the outer loop increment.
// CHECK-LABEL: define {{.*}}void @pr8880_cg_2(
// CHECK:       for.inc:
// CHECK:         br label %for.inc4
// CHECK:       for.inc4:
void pr8880_cg_2(int *iptr) {
  int i, j;
  for (i = 2; i != 10 ; i++ )
    for (j = 3 ; j < 22; (void)({ ++j; continue; j;})) {
      *iptr = 7;
    }
}

// Break in the condition exits the outer loop.
// CHECK-LABEL: define {{.*}}void @pr8880_cg_3(
// CHECK:       for.cond:
// CHECK:       for.cond1:
// CHECK-NEXT:    br label %for.end5
// CHECK:       for.end:
// CHECK:       for.end5:
void pr8880_cg_3(int *iptr) {
  int i, j;
  for (i = 2 ; i != 10 ; i++ )
    for (j = 3 ; ({break; j;}); j++) {
      *iptr = 7;
    }
}

// Continue in the condition targets the outer loop increment.
// CHECK-LABEL: define {{.*}}void @pr8880_cg_4(
// CHECK:       for.cond:
// CHECK:       for.cond1:
// CHECK-NEXT:    br label %for.inc3
// CHECK:       for.inc:
// CHECK:       for.inc3:
void pr8880_cg_4(int *iptr) {
  int i, j;
  for (i = 2 ; i != 10 ; i++ )
    for (j = 3 ; ({continue; j;}); j++) {
      *iptr = 7;
    }
}

// CHECK-LABEL: define {{.*}}void @pr8880_cg_5(
// CHECK:       while.cond:
// CHECK:       while.cond1:
// CHECK-NEXT:    br label %while.end5
// CHECK:       while.end:
// CHECK:       while.end5:
void pr8880_cg_5(int x, int *iptr) {
  int y = 5;
  while(--x) {
    while(({ break; --y; })) {
      *iptr = 7;
    }
  }
}

// CHECK-LABEL: define {{.*}}void @pr8880_cg_6(
// CHECK:       do.cond:
// CHECK-NEXT:    br label %while.end
// CHECK:       do.end:
// CHECK:       while.end:
void pr8880_cg_6(int x, int *iptr) {
  int y = 5;
  while(--x) {
    do {
      *iptr = 7;
    } while(({ break; --y; }));
  }
}

int cond(void);

// CHECK-LABEL: define {{.*}}void @while_continue(
// CHECK:       while.cond:
// CHECK:       while.cond1:
// CHECK-NEXT:    br label %while.cond
void while_continue(void) {
  while (cond())
    while (({ continue; 1; })) {}
}

// CHECK-LABEL: define {{.*}}void @do_continue(
// CHECK:       while.cond:
// CHECK:       do.cond:
// CHECK-NEXT:    br label %while.cond
void do_continue(void) {
  while (cond())
    do {} while (({ continue; 1; }));
}

// CHECK-LABEL: define {{.*}}void @for_init_continue(
// CHECK:       while.cond:
// CHECK:       while.body:
// CHECK-NEXT:    br label %while.cond
void for_init_continue(void) {
  while (cond())
    for (({ continue; });;) {}
}

// Break targets the switch, while continue targets its enclosing loop.
// CHECK-LABEL: define {{.*}}void @switch_break(
// CHECK:       sw.bb:
// CHECK:       for.cond:
// CHECK-NEXT:    br label %sw.epilog
// CHECK:       sw.epilog:
// CHECK:       while.end:
void switch_break(int n) {
  while (cond())
    switch (n) {
    case 0:
      for (; ({ break; 1; });) {}
    }
}

// CHECK-LABEL: define {{.*}}void @switch_continue(
// CHECK:       while.cond:
// CHECK:       sw.bb:
// CHECK:       for.inc:
// CHECK-NEXT:    br label %while.cond
void switch_continue(int n) {
  while (cond())
    switch (n) {
    case 0:
      for (; cond(); ({ continue; })) {}
    }
}

// CHECK-LABEL: define {{.*}}void @loop_in_condition(
// CHECK:       while.cond:
// CHECK:       while.cond1:
// CHECK:       do.body:
// CHECK-NEXT:    br label %do.end
// CHECK:       do.end:
// CHECK-NEXT:    br label %while.cond
void loop_in_condition(void) {
  while (cond())
    while (({ do { break; } while (cond()); continue; 1; })) {}
}

#ifdef __cplusplus
}
#endif

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o - \
// RUN:   | cir-opt --remarks-filter=points-to -verify-diagnostics -o /dev/null \
// RUN:     -pass-pipeline='builtin.module(cir.func(test-cir-points-to))'
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o - \
// RUN:   | cir-opt --remarks-filter-analyse=points-to -o /dev/null 2>&1 \
// RUN:     -pass-pipeline='builtin.module(cir.func(test-cir-points-to))' \
// RUN:   | FileCheck %s --check-prefix=ANALYSIS --implicit-check-not='[Missed]'
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o - \
// RUN:   | cir-opt --remarks-filter-missed=points-to -o /dev/null 2>&1 \
// RUN:     -pass-pipeline='builtin.module(cir.func(test-cir-points-to))' \
// RUN:   | FileCheck %s --check-prefix=MISSED --implicit-check-not='[Analysis]'

// The analysis emits an analysis remark for each load of a pointer from a local
// variable, with the objects the loaded pointer may point to, and a missed
// remark for each object whose address escapes. --remarks-filter=points-to
// enables both kinds, and --remarks-filter-analyse and --remarks-filter-missed
// only one of them.

// ANALYSIS: remark: [Analysis] PointsTo
// MISSED: remark: [Missed] Escaped {{.*}} Remark="the address of p escapes"
// MISSED: remark: [Missed] Escaped {{.*}} Remark="the address of a escapes"

void take(int **);

void two_pointers() {
  int a, b;
  int *p = &a, *q = &b;
  *p = 1; // expected-remark {{points to [a]}}
  *q = 2; // expected-remark {{points to [b]}}
}

void pointer_to_pointer() {
  int a;
  int *p = &a;
  int **pp = &p;
  // expected-remark @below {{points to [p]}}
  // expected-remark @below {{points to [a]}}
  **pp = 1;
}

// A null pointer points to no object.
void null_then_address() {
  int a;
  int *p = nullptr;
  p = &a;
  *p = 1; // expected-remark {{points to [a]}}
}

// The address of `p` is passed to a call, which may store any address into it,
// and can reach `a` through it.
void escape_to_call() {
  int a;
  int *p = &a;
  // expected-remark@+2 {{the address of p escapes}}
  // expected-remark@+1 {{the address of a escapes}}
  take(&p);
  *p = 1; // expected-remark {{points to [a, unknown]}}
}

// `p` is assigned on two paths, so it may point to `a` or `b`.
void two_paths(bool c) {
  int a, b;
  int *p = &a;
  if (c)
    p = &b;
  *p = 1; // expected-remark {{points to [a, b]}}
}

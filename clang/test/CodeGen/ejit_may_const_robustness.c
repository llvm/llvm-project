// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -o - %s | FileCheck %s
//
// EmbeddedJIT may_const metadata robustness (CodeGen level).
//
// Complements:
//   - ejit_metadata.c       (basic period/entry/may_const metadata)
//   - ejit_volatile_field.c (volatile + plain sibling fields are NOT annotated)
//
// New coverage here: repeated loads of the *same* may_const field are EACH
// annotated, and a may_const load reached through a nested struct-field GEP
// carries the annotation. These are the access patterns the JIT-time
// EJitStructFieldPass relies on (see the robustness tests in EJitRuntimeTest.cpp).

struct Inner {
  __attribute__((ejit_may_const)) int v; // nested may_const field
  int pad;
};

struct Cfg {
  __attribute__((ejit_may_const)) int a; // top-level may_const field
  int b;                                 // plain sibling
  struct Inner inner;
};

__attribute__((ejit_period_arr("cell"))) struct Cfg g_cfg[8];

extern void barrier(void);

// Two reads of the SAME may_const field, separated by an opaque call so the
// front end cannot coalesce them into a single load: BOTH must be annotated.
// CHECK-LABEL: define {{.*}}@read_a_twice(
// CHECK: load i32, ptr {{.*}}, !ejit.may_const
// CHECK: load i32, ptr {{.*}}, !ejit.may_const
__attribute__((ejit_entry))
int read_a_twice(__attribute__((ejit_period_arr_ind("cell"))) int ci) {
  int x = g_cfg[ci].a;
  barrier();
  int y = g_cfg[ci].a;
  return x + y;
}

// A may_const field reached through a nested struct GEP is annotated.
// CHECK-LABEL: define {{.*}}@read_nested(
// CHECK: load i32, ptr {{.*}}, !ejit.may_const
__attribute__((ejit_entry))
int read_nested(__attribute__((ejit_period_arr_ind("cell"))) int ci) {
  return g_cfg[ci].inner.v;
}

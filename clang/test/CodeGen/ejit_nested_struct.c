// RUN: %clang_cc1 -emit-llvm -o - %s | FileCheck %s
// Test: Nested struct with ejit_may_const fields generates correct metadata on loads.

struct Inner {
  __attribute__((ejit_may_const)) int b;
  int yyy;
};

struct Outer {
  __attribute__((ejit_may_const)) int a;
  int xxx;
  struct Inner inner;
};

__attribute__((ejit_period_arr("cell"))) struct Outer g_data[8];

// CHECK-DAG: @g_data = {{.*}} !ejit.metadata ![[ARR_META:[0-9]+]]

__attribute__((ejit_entry))
int process_cell(__attribute__((ejit_period_arr_ind("cell"))) int ci) {
  // CHECK: load i32, {{.*}} !ejit.may_const ![[MAYCONST:[0-9]+]]
  int val_a = g_data[ci].a;

  // CHECK: load i32, {{.*}} !ejit.may_const ![[MAYCONST]]
  int val_b = g_data[ci].inner.b;

  // Non-may_const field: NO metadata
  // CHECK-NOT: load i32, {{.*}} !ejit.may_const
  int non_const = g_data[ci].xxx;

  return val_a + val_b + non_const;
}

__attribute__((ejit_entry))
int process_static_only() {
  // Static entries also get may_const loads checked
  return 0;
}

// CHECK-DAG: ![[ARR_META]] = distinct !{![[ARR:[0-9]+]]}
// CHECK-DAG: ![[ARR]] = !{!"ejit_period_arr", !"cell", i32 8}
// CHECK-DAG: ![[MAYCONST]] = !{}

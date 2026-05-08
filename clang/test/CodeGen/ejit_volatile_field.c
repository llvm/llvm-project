// RUN: %clang_cc1 -emit-llvm -o - %s | FileCheck %s
// Test: volatile fields must NOT get !ejit.may_const metadata on loads.

struct Config {
  __attribute__((ejit_may_const)) int cellType;
  __attribute__((ejit_may_const)) volatile int powerLevel; // volatile - no may_const
  int normalField;
};

__attribute__((ejit_period_arr("cell"))) struct Config g_cfg[10];

__attribute__((ejit_entry))
int test_volatile(__attribute__((ejit_period_arr_ind("cell"))) int ci) {
  // CHECK: load i32, {{.*}} !ejit.may_const ![[MAYCONST:[0-9]+]]
  int x = g_cfg[ci].cellType; // may_const applies

  // Volatile field: must NOT get !ejit.may_const
  // CHECK-NOT: load volatile i32, {{.*}} !ejit.may_const
  int y = g_cfg[ci].powerLevel; // volatile, no may_const

  return x + y;
}

// Non-may_const field also gets no metadata
__attribute__((ejit_entry))
int test_normal_field(__attribute__((ejit_period_arr_ind("cell"))) int ci) {
  // Normal field: no may_const metadata
  // CHECK-NOT: load i32, {{.*}} !ejit.may_const
  int z = g_cfg[ci].normalField;
  return z;
}

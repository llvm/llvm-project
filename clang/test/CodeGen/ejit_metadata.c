// RUN: %clang_cc1 -emit-llvm -o - %s | FileCheck %s
// EmbeddedJIT CodeGen metadata tests

struct CellConfig {
  __attribute__((ejit_may_const)) int cellType;
  int xx;
};

// CHECK-DAG: @g_boardCfg = {{.*}} !ejit.metadata ![[PERIOD_META:[0-9]+]]
__attribute__((ejit_period("static"))) struct CellConfig g_boardCfg;

// CHECK-DAG: @g_cellCfg = {{.*}} !ejit.metadata ![[ARR_META:[0-9]+]]
__attribute__((ejit_period_arr("cell"))) struct CellConfig g_cellCfg[16];

// CHECK: define {{.*}}void @jit_entry({{.*}} !ejit.metadata ![[ENTRY_META:[0-9]+]]
__attribute__((ejit_entry))
void jit_entry(__attribute__((ejit_period_arr_ind("cell"))) int cellIdx) {
  // CHECK: load {{.*}} !ejit.may_const ![[MAYCONST:[0-9]+]]
  if (g_cellCfg[cellIdx].cellType == 2) {
    // Access to may_const field should have metadata
  }
}

// CHECK: define {{.*}}void @lc_func({{.*}} !ejit.metadata ![[LC_META:[0-9]+]]
__attribute__((ejit_period_lc("cell")))
void lc_func(__attribute__((ejit_period_arr_ind("cell"))) int cellIdx) {
  g_cellCfg[cellIdx].xx = 42; // modify non-may_const field
}

// CHECK-DAG: ![[PERIOD_META]] = distinct !{![[PERIOD:[0-9]+]]}
// CHECK-DAG: ![[PERIOD]] = !{!"ejit_period", !"static"}
// CHECK-DAG: ![[ARR_META]] = distinct !{![[ARR:[0-9]+]]}
// CHECK-DAG: ![[ARR]] = !{!"ejit_period_arr", !"cell", i32 16}
// CHECK-DAG: ![[ENTRY_META]] = distinct !{![[ENTRY:[0-9]+]], ![[IND:[0-9]+]]}
// CHECK-DAG: ![[ENTRY]] = !{!"ejit_entry"}
// CHECK-DAG: ![[LC_META]] = distinct !{![[LC:[0-9]+]], ![[IND]]}
// CHECK-DAG: ![[LC]] = !{!"ejit_period_lc", !"cell"}
// CHECK-DAG: ![[IND]] = !{!"ejit_period_arr_ind", !"cell", i32 0}
// CHECK-DAG: ![[MAYCONST]] = !{}

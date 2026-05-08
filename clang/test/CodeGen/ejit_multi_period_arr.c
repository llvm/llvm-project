// RUN: %clang_cc1 -emit-llvm -o - %s | FileCheck %s
// Test: Multiple arrays sharing the same period name "cell".

struct CellConfig {
  __attribute__((ejit_may_const)) int cellType;
  int trafficLoad;
};

struct CellPhy {
  __attribute__((ejit_may_const)) int phyCellId;
  int rssi;
};

// CHECK-DAG: @cellCfg = {{.*}} !ejit.metadata ![[CELL_META:[0-9]+]]
__attribute__((ejit_period_arr("cell"))) struct CellConfig cellCfg[16];

// CHECK-DAG: @cellPhy = {{.*}} !ejit.metadata ![[PHY_META:[0-9]+]]
__attribute__((ejit_period_arr("cell"))) struct CellPhy cellPhy[16];

// CHECK-DAG: @boardCfg = {{.*}} !ejit.metadata ![[BOARD_META:[0-9]+]]
__attribute__((ejit_period("static"))) int boardCfg;

__attribute__((ejit_entry))
int process_cell(__attribute__((ejit_period_arr_ind("cell"))) int ci) {
  // CHECK: load i32, {{.*}} !ejit.may_const ![[MAYCONST:[0-9]+]]
  int cell_type = cellCfg[ci].cellType;

  // CHECK: load i32, {{.*}} !ejit.may_const ![[MAYCONST]]
  int phy_id = cellPhy[ci].phyCellId;

  // boardCfg is a scalar int, not a struct field with may_const
  int board = boardCfg;

  return cell_type + phy_id + board;
}

// CHECK-DAG: ![[MAYCONST]] = !{}

// Period metadata: both arrays use period name "cell" (may share inner node)
// CHECK-DAG: ![[CELL_INNER:[0-9]+]] = !{!"ejit_period_arr", !"cell", i32 16}
// CHECK-DAG: ![[BOARD_INNER:[0-9]+]] = !{!"ejit_period", !"static"}

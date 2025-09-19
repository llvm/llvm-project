// RUN: fir-opt --lower-workdistribute %s | FileCheck %s
// Test lowering of workdistribute after fission on host device.

// CHECK-LABEL:   func.func @x(
// CHECK:           %[[VAL_0:.*]] = fir.alloca index {bindc_name = "lb"}
// CHECK:           fir.store %[[ARG0:.*]] to %[[VAL_0]] : !fir.ref<index>
// CHECK:           %[[VAL_1:.*]] = fir.alloca index {bindc_name = "ub"}
// CHECK:           fir.store %[[ARG1:.*]] to %[[VAL_1]] : !fir.ref<index>
// CHECK:           %[[VAL_2:.*]] = fir.alloca index {bindc_name = "step"}
// CHECK:           fir.store %[[ARG2:.*]] to %[[VAL_2]] : !fir.ref<index>
// CHECK:           %[[VAL_3:.*]] = omp.map.info var_ptr(%[[VAL_0]] : !fir.ref<index>, index) map_clauses(to) capture(ByRef) -> !fir.ref<index> {name = "lb"}
// CHECK:           %[[VAL_4:.*]] = omp.map.info var_ptr(%[[VAL_1]] : !fir.ref<index>, index) map_clauses(to) capture(ByRef) -> !fir.ref<index> {name = "ub"}
// CHECK:           %[[VAL_5:.*]] = omp.map.info var_ptr(%[[VAL_2]] : !fir.ref<index>, index) map_clauses(to) capture(ByRef) -> !fir.ref<index> {name = "step"}
// CHECK:           %[[VAL_6:.*]] = omp.map.info var_ptr(%[[ARG3:.*]] : !fir.ref<index>, index) map_clauses(tofrom) capture(ByRef) -> !fir.ref<index> {name = "addr"}
// CHECK:           %[[VAL_7:.*]] = omp.map.info var_ptr(%[[VAL_0]] : !fir.ref<index>, index) map_clauses(exit_release_or_enter_alloc) capture(ByRef) -> !fir.ref<index> {name = "lb"}
// CHECK:           %[[VAL_8:.*]] = omp.map.info var_ptr(%[[VAL_1]] : !fir.ref<index>, index) map_clauses(exit_release_or_enter_alloc) capture(ByRef) -> !fir.ref<index> {name = "ub"}
// CHECK:           %[[VAL_9:.*]] = omp.map.info var_ptr(%[[VAL_2]] : !fir.ref<index>, index) map_clauses(exit_release_or_enter_alloc) capture(ByRef) -> !fir.ref<index> {name = "step"}
// CHECK:           %[[VAL_10:.*]] = omp.map.info var_ptr(%[[ARG3]] : !fir.ref<index>, index) map_clauses(exit_release_or_enter_alloc) capture(ByRef) -> !fir.ref<index> {name = "addr"}
// CHECK:           omp.target_data map_entries(%[[VAL_3]], %[[VAL_4]], %[[VAL_5]], %[[VAL_6]] : !fir.ref<index>, !fir.ref<index>, !fir.ref<index>, !fir.ref<index>) {
// CHECK:             %[[VAL_11:.*]] = fir.alloca index
// CHECK:             %[[VAL_12:.*]] = omp.map.info var_ptr(%[[VAL_11]] : !fir.ref<index>, index) map_clauses(from) capture(ByRef) -> !fir.ref<index> {name = "__flang_workdistribute_from"}
// CHECK:             %[[VAL_13:.*]] = omp.map.info var_ptr(%[[VAL_11]] : !fir.ref<index>, index) map_clauses(to) capture(ByRef) -> !fir.ref<index> {name = "__flang_workdistribute_to"}
// CHECK:             %[[VAL_14:.*]] = fir.alloca index
// CHECK:             %[[VAL_15:.*]] = omp.map.info var_ptr(%[[VAL_14]] : !fir.ref<index>, index) map_clauses(from) capture(ByRef) -> !fir.ref<index> {name = "__flang_workdistribute_from"}
// CHECK:             %[[VAL_16:.*]] = omp.map.info var_ptr(%[[VAL_14]] : !fir.ref<index>, index) map_clauses(to) capture(ByRef) -> !fir.ref<index> {name = "__flang_workdistribute_to"}
// CHECK:             %[[VAL_17:.*]] = fir.alloca index
// CHECK:             %[[VAL_18:.*]] = omp.map.info var_ptr(%[[VAL_17]] : !fir.ref<index>, index) map_clauses(from) capture(ByRef) -> !fir.ref<index> {name = "__flang_workdistribute_from"}
// CHECK:             %[[VAL_19:.*]] = omp.map.info var_ptr(%[[VAL_17]] : !fir.ref<index>, index) map_clauses(to) capture(ByRef) -> !fir.ref<index> {name = "__flang_workdistribute_to"}
// CHECK:             %[[VAL_20:.*]] = fir.alloca !fir.heap<index>
// CHECK:             %[[VAL_21:.*]] = omp.map.info var_ptr(%[[VAL_20]] : !fir.ref<!fir.heap<index>>, !fir.heap<index>) map_clauses(from) capture(ByRef) -> !fir.ref<!fir.heap<index>> {name = "__flang_workdistribute_from"}
// CHECK:             %[[VAL_22:.*]] = omp.map.info var_ptr(%[[VAL_20]] : !fir.ref<!fir.heap<index>>, !fir.heap<index>) map_clauses(to) capture(ByRef) -> !fir.ref<!fir.heap<index>> {name = "__flang_workdistribute_to"}
// CHECK:             %[[VAL_23:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:             %[[VAL_24:.*]] = fir.load %[[VAL_0]] : !fir.ref<index>
// CHECK:             %[[VAL_25:.*]] = fir.load %[[VAL_1]] : !fir.ref<index>
// CHECK:             %[[VAL_26:.*]] = fir.load %[[VAL_2]] : !fir.ref<index>
// CHECK:             %[[VAL_27:.*]] = arith.constant 1 : index
// CHECK:             %[[VAL_28:.*]] = arith.addi %[[VAL_25]], %[[VAL_25]] : index
// CHECK:             %[[VAL_29:.*]] = omp.target_allocmem %[[VAL_23]] : i32, index, %[[VAL_27]] {uniq_name = "dev_buf"}
// CHECK:             %[[VAL_30:.*]] = fir.convert %[[VAL_29]] : (i64) -> !fir.heap<index>
// CHECK:             fir.store %[[VAL_24]] to %[[VAL_11]] : !fir.ref<index>
// CHECK:             fir.store %[[VAL_25]] to %[[VAL_14]] : !fir.ref<index>
// CHECK:             fir.store %[[VAL_26]] to %[[VAL_17]] : !fir.ref<index>
// CHECK:             fir.store %[[VAL_30]] to %[[VAL_20]] : !fir.ref<!fir.heap<index>>
// CHECK:             omp.target map_entries(%[[VAL_7]] -> %[[VAL_31:.*]], %[[VAL_8]] -> %[[VAL_32:.*]], %[[VAL_9]] -> %[[VAL_33:.*]], %[[VAL_10]] -> %[[VAL_34:.*]], %[[VAL_13]] -> %[[VAL_35:.*]], %[[VAL_16]] -> %[[VAL_36:.*]], %[[VAL_19]] -> %[[VAL_37:.*]], %[[VAL_22]] -> %[[VAL_38:.*]] : !fir.ref<index>, !fir.ref<index>, !fir.ref<index>, !fir.ref<index>, !fir.ref<index>, !fir.ref<index>, !fir.ref<index>, !fir.ref<!fir.heap<index>>) {
// CHECK:               %[[VAL_39:.*]] = fir.load %[[VAL_35]] : !fir.ref<index>
// CHECK:               %[[VAL_40:.*]] = fir.load %[[VAL_36]] : !fir.ref<index>
// CHECK:               %[[VAL_41:.*]] = fir.load %[[VAL_37]] : !fir.ref<index>
// CHECK:               %[[VAL_42:.*]] = fir.load %[[VAL_38]] : !fir.ref<!fir.heap<index>>
// CHECK:               %[[VAL_43:.*]] = arith.addi %[[VAL_40]], %[[VAL_40]] : index
// CHECK:               omp.teams {
// CHECK:                 omp.parallel {
// CHECK:                   omp.distribute {
// CHECK:                     omp.wsloop {
// CHECK:                       omp.loop_nest (%[[VAL_44:.*]]) : index = (%[[VAL_39]]) to (%[[VAL_40]]) inclusive step (%[[VAL_41]]) {
// CHECK:                         fir.store %[[VAL_43]] to %[[VAL_42]] : !fir.heap<index>
// CHECK:                         omp.yield
// CHECK:                       }
// CHECK:                     } {omp.composite}
// CHECK:                   } {omp.composite}
// CHECK:                   omp.terminator
// CHECK:                 } {omp.composite}
// CHECK:                 omp.terminator
// CHECK:               }
// CHECK:               omp.terminator
// CHECK:             }
// CHECK:             %[[VAL_45:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:             %[[VAL_46:.*]] = fir.load %[[VAL_11]] : !fir.ref<index>
// CHECK:             %[[VAL_47:.*]] = fir.load %[[VAL_14]] : !fir.ref<index>
// CHECK:             %[[VAL_48:.*]] = fir.load %[[VAL_17]] : !fir.ref<index>
// CHECK:             %[[VAL_49:.*]] = fir.load %[[VAL_20]] : !fir.ref<!fir.heap<index>>
// CHECK:             %[[VAL_50:.*]] = arith.addi %[[VAL_47]], %[[VAL_47]] : index
// CHECK:             fir.store %[[VAL_46]] to %[[VAL_49]] : !fir.heap<index>
// CHECK:             %[[VAL_51:.*]] = fir.convert %[[VAL_49]] : (!fir.heap<index>) -> i64
// CHECK:             omp.target_freemem %[[VAL_45]], %[[VAL_51]] : i32, i64
// CHECK:             omp.terminator
// CHECK:           }
// CHECK:           return
// CHECK:         }


module attributes {llvm.target_triple = "amdgcn-amd-amdhsa", omp.is_gpu = true, omp.is_target_device = true} {
func.func @x(%lb : index, %ub : index, %step : index, %addr : !fir.ref<index>) {
  %lb_ref = fir.alloca index {bindc_name = "lb"}
  fir.store %lb to %lb_ref : !fir.ref<index>
  %ub_ref = fir.alloca index {bindc_name = "ub"}
  fir.store %ub to %ub_ref : !fir.ref<index>
  %step_ref = fir.alloca index {bindc_name = "step"}
  fir.store %step to %step_ref : !fir.ref<index>

  %lb_map = omp.map.info var_ptr(%lb_ref : !fir.ref<index>, index) map_clauses(to) capture(ByRef) -> !fir.ref<index> {name = "lb"}
  %ub_map = omp.map.info var_ptr(%ub_ref : !fir.ref<index>, index) map_clauses(to) capture(ByRef) -> !fir.ref<index> {name = "ub"}
  %step_map = omp.map.info var_ptr(%step_ref : !fir.ref<index>, index) map_clauses(to) capture(ByRef) -> !fir.ref<index> {name = "step"}
  %addr_map = omp.map.info var_ptr(%addr : !fir.ref<index>, index) map_clauses(tofrom) capture(ByRef) -> !fir.ref<index> {name = "addr"}

  omp.target map_entries(%lb_map -> %ARG0, %ub_map -> %ARG1, %step_map -> %ARG2, %addr_map -> %ARG3 : !fir.ref<index>, !fir.ref<index>, !fir.ref<index>, !fir.ref<index>) {
    %lb_val = fir.load %ARG0 : !fir.ref<index>
    %ub_val = fir.load %ARG1 : !fir.ref<index>
    %step_val = fir.load %ARG2 : !fir.ref<index>
    %one = arith.constant 1 : index

    %20 = arith.addi %ub_val, %ub_val : index
    omp.teams {
      omp.workdistribute {
        %dev_mem = fir.allocmem index, %one {uniq_name = "dev_buf"}
        fir.do_loop %iv = %lb_val to %ub_val step %step_val unordered {
          fir.store %20 to %dev_mem : !fir.heap<index>
        }
        fir.store %lb_val to %dev_mem : !fir.heap<index>
        fir.freemem %dev_mem : !fir.heap<index>
        omp.terminator
      }
      omp.terminator
    }
    omp.terminator
  }
  return
}
}

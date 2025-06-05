// RUN: fir-opt --lower-workdistribute %s | FileCheck %s

// CHECK-LABEL:   func.func @x
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
// CHECK:           omp.target_data map_entries(%[[VAL_3]], %[[VAL_4]], %[[VAL_5]], %[[VAL_6]] : !fir.ref<index>, !fir.ref<index>, !fir.ref<index>, !fir.ref<index>) {
// CHECK:             %[[VAL_7:.*]] = omp.map.info var_ptr(%[[VAL_0]] : !fir.ref<index>, index) map_clauses(to) capture(ByRef) -> !fir.ref<index> {name = "lb"}
// CHECK:             %[[VAL_8:.*]] = omp.map.info var_ptr(%[[VAL_1]] : !fir.ref<index>, index) map_clauses(to) capture(ByRef) -> !fir.ref<index> {name = "ub"}
// CHECK:             %[[VAL_9:.*]] = omp.map.info var_ptr(%[[VAL_2]] : !fir.ref<index>, index) map_clauses(to) capture(ByRef) -> !fir.ref<index> {name = "step"}
// CHECK:             %[[VAL_10:.*]] = omp.map.info var_ptr(%[[ARG3:.*]] : !fir.ref<index>, index) map_clauses(tofrom) capture(ByRef) -> !fir.ref<index> {name = "addr"}
// CHECK:             %[[VAL_11:.*]] = fir.alloca !fir.heap<index>
// CHECK:             %[[VAL_12:.*]] = omp.map.info var_ptr(%[[VAL_11]] : !fir.ref<!fir.heap<index>>, !fir.heap<index>) map_clauses(from) capture(ByRef) -> !fir.ref<!fir.heap<index>> {name = "__flang_workdistribute_from"}
// CHECK:             %[[VAL_13:.*]] = omp.map.info var_ptr(%[[VAL_11]] : !fir.ref<!fir.heap<index>>, !fir.heap<index>) map_clauses(to) capture(ByRef) -> !fir.ref<!fir.heap<index>> {name = "__flang_workdistribute_to"}
// CHECK:             omp.target map_entries(%[[VAL_7]] -> %[[VAL_14:.*]], %[[VAL_8]] -> %[[VAL_15:.*]], %[[VAL_9]] -> %[[VAL_16:.*]], %[[VAL_10]] -> %[[VAL_17:.*]], %[[VAL_12]] -> %[[VAL_18:.*]] : !fir.ref<index>, !fir.ref<index>, !fir.ref<index>, !fir.ref<index>, !fir.ref<!fir.heap<index>>) {
// CHECK:               %[[VAL_19:.*]] = arith.constant 1 : index
// CHECK:               %[[VAL_20:.*]] = fir.load %[[VAL_14]] : !fir.ref<index>
// CHECK:               %[[VAL_21:.*]] = fir.load %[[VAL_15]] : !fir.ref<index>
// CHECK:               %[[VAL_22:.*]] = fir.load %[[VAL_16]] : !fir.ref<index>
// CHECK:               %[[VAL_23:.*]] = arith.addi %[[VAL_21]], %[[VAL_21]] : index
// CHECK:               %[[VAL_24:.*]] = fir.allocmem index, %[[VAL_19]] {uniq_name = "dev_buf"}
// CHECK:               fir.store %[[VAL_24]] to %[[VAL_18]] : !fir.llvm_ptr<!fir.heap<index>>
// CHECK:               omp.terminator
// CHECK:             }
// CHECK:             omp.target map_entries(%[[VAL_7]] -> %[[VAL_25:.*]], %[[VAL_8]] -> %[[VAL_26:.*]], %[[VAL_9]] -> %[[VAL_27:.*]], %[[VAL_10]] -> %[[VAL_28:.*]], %[[VAL_13]] -> %[[VAL_29:.*]] : !fir.ref<index>, !fir.ref<index>, !fir.ref<index>, !fir.ref<index>, !fir.ref<!fir.heap<index>>) {
// CHECK:               %[[VAL_30:.*]] = fir.load %[[VAL_29]] : !fir.llvm_ptr<!fir.heap<index>>
// CHECK:               %[[VAL_31:.*]] = fir.load %[[VAL_25]] : !fir.ref<index>
// CHECK:               %[[VAL_32:.*]] = fir.load %[[VAL_26]] : !fir.ref<index>
// CHECK:               %[[VAL_33:.*]] = fir.load %[[VAL_27]] : !fir.ref<index>
// CHECK:               %[[VAL_34:.*]] = arith.addi %[[VAL_32]], %[[VAL_32]] : index
// CHECK:               omp.teams {
// CHECK:                 omp.parallel {
// CHECK:                   omp.distribute {
// CHECK:                     omp.wsloop {
// CHECK:                       omp.loop_nest (%[[VAL_35:.*]]) : index = (%[[VAL_31]]) to (%[[VAL_32]]) inclusive step (%[[VAL_33]]) {
// CHECK:                         fir.store %[[VAL_34]] to %[[VAL_30]] : !fir.heap<index>
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
// CHECK:             omp.target map_entries(%[[VAL_7]] -> %[[VAL_36:.*]], %[[VAL_8]] -> %[[VAL_37:.*]], %[[VAL_9]] -> %[[VAL_38:.*]], %[[VAL_10]] -> %[[VAL_39:.*]], %[[VAL_13]] -> %[[VAL_40:.*]] : !fir.ref<index>, !fir.ref<index>, !fir.ref<index>, !fir.ref<index>, !fir.ref<!fir.heap<index>>) {
// CHECK:               %[[VAL_41:.*]] = fir.load %[[VAL_40]] : !fir.llvm_ptr<!fir.heap<index>>
// CHECK:               %[[VAL_42:.*]] = fir.load %[[VAL_36]] : !fir.ref<index>
// CHECK:               %[[VAL_43:.*]] = fir.load %[[VAL_37]] : !fir.ref<index>
// CHECK:               %[[VAL_44:.*]] = fir.load %[[VAL_38]] : !fir.ref<index>
// CHECK:               %[[VAL_45:.*]] = arith.addi %[[VAL_43]], %[[VAL_43]] : index
// CHECK:               fir.store %[[VAL_42]] to %[[VAL_41]] : !fir.heap<index>
// CHECK:               fir.freemem %[[VAL_41]] : !fir.heap<index>
// CHECK:               omp.terminator
// CHECK:             }
// CHECK:             omp.terminator
// CHECK:           }
// CHECK:           return
// CHECK:         }

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

  omp.target map_entries(%lb_map -> %arg0, %ub_map -> %arg1, %step_map -> %arg2, %addr_map -> %arg3 : !fir.ref<index>, !fir.ref<index>, !fir.ref<index>, !fir.ref<index>) {
    %lb_val = fir.load %arg0 : !fir.ref<index>
    %ub_val = fir.load %arg1 : !fir.ref<index>
    %step_val = fir.load %arg2 : !fir.ref<index>
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

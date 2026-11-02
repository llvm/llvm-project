// RUN: fir-opt --lower-workdistribute %s | FileCheck %s

// A Fortran allocatable descriptor (fir.box) crossing the workdistribute target
// fission must be recomputed inside the isolated target from its mapped
// descriptor, not cached by value via __flang_workdistribute_to/from. Caching
// the box would freeze a host base_addr into the device kernel.

// The kernel reloads the box from its own mapped descriptor argument.
// CHECK-LABEL:   func.func @recompute_descriptor(
// CHECK-SAME:      %[[X:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) {
// CHECK:           %[[X_MAP:.*]] = omp.map.info var_ptr(%[[X]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>, !fir.box<!fir.heap<!fir.array<?xf32>>>) map_clauses(tofrom) capture(ByRef) name("x") -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
// CHECK:           %[[X_STORAGE:.*]] = omp.map.info var_ptr(%[[X]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>, !fir.box<!fir.heap<!fir.array<?xf32>>>) map_clauses(storage) capture(ByRef) name("x") -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
// CHECK:           omp.target_data map_entries(%[[X_MAP]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) {
// CHECK-NOT:         __flang_workdistribute
// CHECK:             %[[MLIR_0:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:             %[[CONSTANT_0:.*]] = arith.constant 0 : index
// CHECK:             %[[CONSTANT_1:.*]] = arith.constant 1 : index
// CHECK:             %[[CONSTANT_2:.*]] = arith.constant 9 : index
// CHECK:             %[[CONSTANT_3:.*]] = arith.constant 5.000000e+00 : f32
// CHECK:             %[[LOAD_0:.*]] = fir.load %[[X]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
// CHECK:             omp.target kernel_type(spmd) map_entries(%[[X_STORAGE]] -> %[[X_ARG:.*]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) {
// CHECK:               %[[CONSTANT_4:.*]] = arith.constant 0 : index
// CHECK:               %[[CONSTANT_5:.*]] = arith.constant 1 : index
// CHECK:               %[[CONSTANT_6:.*]] = arith.constant 9 : index
// CHECK:               %[[CONSTANT_7:.*]] = arith.constant 5.000000e+00 : f32
// CHECK:               %[[BOX:.*]] = fir.load %[[X_ARG]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
// CHECK:               omp.teams {
// CHECK:                 omp.parallel {
// CHECK:                   omp.distribute {
// CHECK:                     omp.wsloop {
// CHECK:                       omp.loop_nest (%[[IV:.*]]) : index = (%[[CONSTANT_4]]) to (%[[CONSTANT_6]]) inclusive step (%[[CONSTANT_5]]) {
// CHECK:                         %[[BOX_ADDR_0:.*]] = fir.box_addr %[[BOX]] : (!fir.box<!fir.heap<!fir.array<?xf32>>>) -> !fir.heap<!fir.array<?xf32>>
// CHECK:                         %[[COORDINATE_OF_0:.*]] = fir.coordinate_of %[[BOX_ADDR_0]], %[[IV]] : (!fir.heap<!fir.array<?xf32>>, index) -> !fir.ref<f32>
// CHECK:                         fir.store %[[CONSTANT_7]] to %[[COORDINATE_OF_0]] : !fir.ref<f32>
// CHECK:                         omp.yield
// CHECK:                       }
// CHECK:                     } {omp.composite}
// CHECK:                   } {omp.composite}
// CHECK:                   omp.terminator
// CHECK:                 } {omp.composite}
// CHECK:                 omp.terminator
// CHECK:               } {omp.combined}
// CHECK:               omp.terminator
// CHECK:             } {omp.combined}
// CHECK:             omp.terminator
// CHECK:           }
// CHECK:           return
// CHECK:         }

module attributes {llvm.target_triple = "amdgcn-amd-amdhsa", omp.is_gpu = true, omp.is_target_device = true} {
func.func @recompute_descriptor(%arg0: !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) {
  %map = omp.map.info var_ptr(%arg0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>, !fir.box<!fir.heap<!fir.array<?xf32>>>) map_clauses(tofrom) capture(ByRef) name("x") -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  omp.target kernel_type(generic) map_entries(%map -> %barg : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c9 = arith.constant 9 : index
    %cst = arith.constant 5.000000e+00 : f32
    // The descriptor load crosses the split - it must be recomputed, not cached.
    %box = fir.load %barg : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
    omp.teams {
      omp.workdistribute {
        fir.do_loop %iv = %c0 to %c9 step %c1 unordered {
          %addr = fir.box_addr %box : (!fir.box<!fir.heap<!fir.array<?xf32>>>) -> !fir.heap<!fir.array<?xf32>>
          %coor = fir.coordinate_of %addr, %iv : (!fir.heap<!fir.array<?xf32>>, index) -> !fir.ref<f32>
          fir.store %cst to %coor : !fir.ref<f32>
        }
        omp.terminator
      }
      omp.terminator
    }
    omp.terminator
  }
  return
}

// A store before the load, like the realloc of an allocatable LHS, does not change what a re-read sees.
// CHECK-LABEL:   func.func @recompute_after_store(
// CHECK-SAME:      %[[X:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>,
// CHECK-SAME:      %[[Y:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) {
// CHECK:           %[[X_MAP:.*]] = omp.map.info var_ptr(%[[X]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>, !fir.box<!fir.heap<!fir.array<?xf32>>>) map_clauses(tofrom) capture(ByRef) name("x") -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
// CHECK:           %[[Y_MAP:.*]] = omp.map.info var_ptr(%[[Y]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>, !fir.box<!fir.heap<!fir.array<?xf32>>>) map_clauses(tofrom) capture(ByRef) name("y") -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
// CHECK:           %[[X_STORAGE:.*]] = omp.map.info var_ptr(%[[X]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>, !fir.box<!fir.heap<!fir.array<?xf32>>>) map_clauses(storage) capture(ByRef) name("x") -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
// CHECK:           %[[Y_STORAGE:.*]] = omp.map.info var_ptr(%[[Y]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>, !fir.box<!fir.heap<!fir.array<?xf32>>>) map_clauses(storage) capture(ByRef) name("y") -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
// CHECK:           omp.target_data map_entries(%[[X_MAP]], %[[Y_MAP]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) {
// CHECK-NOT:         __flang_workdistribute
// CHECK:             %[[MLIR_0:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:             %[[CONSTANT_0:.*]] = arith.constant 0 : index
// CHECK:             %[[CONSTANT_1:.*]] = arith.constant 1 : index
// CHECK:             %[[CONSTANT_2:.*]] = arith.constant 9 : index
// CHECK:             %[[CONSTANT_3:.*]] = arith.constant 5.000000e+00 : f32
// CHECK:             %[[OTHER:.*]] = fir.load %[[Y]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
// CHECK:             fir.store %[[OTHER]] to %[[X]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
// CHECK:             %[[LOAD_1:.*]] = fir.load %[[X]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
// CHECK:             omp.target kernel_type(spmd) map_entries(%[[X_STORAGE]] -> %[[X_ARG:.*]], %[[Y_STORAGE]] -> %[[Y_ARG:.*]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) {
// CHECK:               %[[CONSTANT_4:.*]] = arith.constant 0 : index
// CHECK:               %[[CONSTANT_5:.*]] = arith.constant 1 : index
// CHECK:               %[[CONSTANT_6:.*]] = arith.constant 9 : index
// CHECK:               %[[CONSTANT_7:.*]] = arith.constant 5.000000e+00 : f32
// CHECK:               %[[BOX:.*]] = fir.load %[[X_ARG]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
// CHECK:               omp.teams {
// CHECK:                 omp.parallel {
// CHECK:                   omp.distribute {
// CHECK:                     omp.wsloop {
// CHECK:                       omp.loop_nest (%[[IV:.*]]) : index = (%[[CONSTANT_4]]) to (%[[CONSTANT_6]]) inclusive step (%[[CONSTANT_5]]) {
// CHECK:                         %[[BOX_ADDR_0:.*]] = fir.box_addr %[[BOX]] : (!fir.box<!fir.heap<!fir.array<?xf32>>>) -> !fir.heap<!fir.array<?xf32>>
// CHECK:                         %[[COORDINATE_OF_0:.*]] = fir.coordinate_of %[[BOX_ADDR_0]], %[[IV]] : (!fir.heap<!fir.array<?xf32>>, index) -> !fir.ref<f32>
// CHECK:                         fir.store %[[CONSTANT_7]] to %[[COORDINATE_OF_0]] : !fir.ref<f32>
// CHECK:                         omp.yield
// CHECK:                       }
// CHECK:                     } {omp.composite}
// CHECK:                   } {omp.composite}
// CHECK:                   omp.terminator
// CHECK:                 } {omp.composite}
// CHECK:                 omp.terminator
// CHECK:               } {omp.combined}
// CHECK:               omp.terminator
// CHECK:             } {omp.combined}
// CHECK:             omp.terminator
// CHECK:           }
// CHECK:           return
// CHECK:         }
func.func @recompute_after_store(%arg0: !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>, %arg1: !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) {
  %map0 = omp.map.info var_ptr(%arg0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>, !fir.box<!fir.heap<!fir.array<?xf32>>>) map_clauses(tofrom) capture(ByRef) name("x") -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  %map1 = omp.map.info var_ptr(%arg1 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>, !fir.box<!fir.heap<!fir.array<?xf32>>>) map_clauses(tofrom) capture(ByRef) name("y") -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  omp.target kernel_type(generic) map_entries(%map0 -> %bx, %map1 -> %by : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c9 = arith.constant 9 : index
    %cst = arith.constant 5.000000e+00 : f32
    %other = fir.load %by : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
    fir.store %other to %bx : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
    %box = fir.load %bx : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
    omp.teams {
      omp.workdistribute {
        fir.do_loop %iv = %c0 to %c9 step %c1 unordered {
          %addr = fir.box_addr %box : (!fir.box<!fir.heap<!fir.array<?xf32>>>) -> !fir.heap<!fir.array<?xf32>>
          %coor = fir.coordinate_of %addr, %iv : (!fir.heap<!fir.array<?xf32>>, index) -> !fir.ref<f32>
          fir.store %cst to %coor : !fir.ref<f32>
        }
        omp.terminator
      }
      omp.terminator
    }
    omp.terminator
  }
  return
}

// A box loaded from a local copy must be cached, since the initializing store is not cloned.
// CHECK-LABEL:   func.func @cache_local_descriptor(
// CHECK-SAME:      %[[X:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) {
// CHECK:           %[[X_MAP:.*]] = omp.map.info var_ptr(%[[X]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>, !fir.box<!fir.heap<!fir.array<?xf32>>>) map_clauses(tofrom) capture(ByRef) name("x") -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
// CHECK:           %[[X_STORAGE:.*]] = omp.map.info var_ptr(%[[X]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>, !fir.box<!fir.heap<!fir.array<?xf32>>>) map_clauses(storage) capture(ByRef) name("x") -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
// CHECK:           omp.target_data map_entries(%[[X_MAP]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) {
// CHECK:             %[[CACHE:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xf32>>>
// CHECK:             %[[CACHE_FROM:.*]] = omp.map.info var_ptr(%[[CACHE]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>, !fir.box<!fir.heap<!fir.array<?xf32>>>) map_clauses(from) capture(ByRef) name("__flang_workdistribute_from") -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
// CHECK:             %[[CACHE_TO:.*]] = omp.map.info var_ptr(%[[CACHE]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>, !fir.box<!fir.heap<!fir.array<?xf32>>>) map_clauses(to) capture(ByRef) name("__flang_workdistribute_to") -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
// CHECK:             %[[MLIR_0:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:             %[[CONSTANT_0:.*]] = arith.constant 0 : index
// CHECK:             %[[CONSTANT_1:.*]] = arith.constant 1 : index
// CHECK:             %[[CONSTANT_2:.*]] = arith.constant 9 : index
// CHECK:             %[[CONSTANT_3:.*]] = arith.constant 5.000000e+00 : f32
// CHECK:             %[[LOCAL:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xf32>>>
// CHECK:             %[[INIT:.*]] = fir.load %[[X]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
// CHECK:             fir.store %[[INIT]] to %[[LOCAL]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
// CHECK:             %[[LOCAL_BOX:.*]] = fir.load %[[LOCAL]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
// CHECK:             fir.store %[[LOCAL_BOX]] to %[[CACHE]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
// CHECK:             omp.target kernel_type(spmd) map_entries(%[[X_STORAGE]] -> %[[X_ARG:.*]], %[[CACHE_TO]] -> %[[CACHE_ARG:.*]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) {
// CHECK:               %[[BOX:.*]] = fir.load %[[CACHE_ARG]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
// CHECK:               %[[CONSTANT_4:.*]] = arith.constant 0 : index
// CHECK:               %[[CONSTANT_5:.*]] = arith.constant 1 : index
// CHECK:               %[[CONSTANT_6:.*]] = arith.constant 9 : index
// CHECK:               %[[CONSTANT_7:.*]] = arith.constant 5.000000e+00 : f32
// CHECK:               omp.teams {
// CHECK:                 omp.parallel {
// CHECK:                   omp.distribute {
// CHECK:                     omp.wsloop {
// CHECK:                       omp.loop_nest (%[[IV:.*]]) : index = (%[[CONSTANT_4]]) to (%[[CONSTANT_6]]) inclusive step (%[[CONSTANT_5]]) {
// CHECK:                         %[[BOX_ADDR_0:.*]] = fir.box_addr %[[BOX]] : (!fir.box<!fir.heap<!fir.array<?xf32>>>) -> !fir.heap<!fir.array<?xf32>>
// CHECK:                         %[[COORDINATE_OF_0:.*]] = fir.coordinate_of %[[BOX_ADDR_0]], %[[IV]] : (!fir.heap<!fir.array<?xf32>>, index) -> !fir.ref<f32>
// CHECK:                         fir.store %[[CONSTANT_7]] to %[[COORDINATE_OF_0]] : !fir.ref<f32>
// CHECK:                         omp.yield
// CHECK:                       }
// CHECK:                     } {omp.composite}
// CHECK:                   } {omp.composite}
// CHECK:                   omp.terminator
// CHECK:                 } {omp.composite}
// CHECK:                 omp.terminator
// CHECK:               } {omp.combined}
// CHECK:               omp.terminator
// CHECK:             } {omp.combined}
// CHECK:             omp.terminator
// CHECK:           }
// CHECK:           return
// CHECK:         }
func.func @cache_local_descriptor(%arg0: !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) {
  %map = omp.map.info var_ptr(%arg0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>, !fir.box<!fir.heap<!fir.array<?xf32>>>) map_clauses(tofrom) capture(ByRef) name("x") -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  omp.target kernel_type(generic) map_entries(%map -> %barg : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c9 = arith.constant 9 : index
    %cst = arith.constant 5.000000e+00 : f32
    %local = fir.alloca !fir.box<!fir.heap<!fir.array<?xf32>>>
    %init = fir.load %barg : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
    fir.store %init to %local : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
    %box = fir.load %local : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
    omp.teams {
      omp.workdistribute {
        fir.do_loop %iv = %c0 to %c9 step %c1 unordered {
          %addr = fir.box_addr %box : (!fir.box<!fir.heap<!fir.array<?xf32>>>) -> !fir.heap<!fir.array<?xf32>>
          %coor = fir.coordinate_of %addr, %iv : (!fir.heap<!fir.array<?xf32>>, index) -> !fir.ref<f32>
          fir.store %cst to %coor : !fir.ref<f32>
        }
        omp.terminator
      }
      omp.terminator
    }
    omp.terminator
  }
  return
}

// A mapped box that is stored to after the load must be cached to keep the loaded value.
// CHECK-LABEL:   func.func @cache_written_descriptor(
// CHECK-SAME:      %[[X:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>,
// CHECK-SAME:      %[[Y:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) {
// CHECK:           %[[X_MAP:.*]] = omp.map.info var_ptr(%[[X]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>, !fir.box<!fir.heap<!fir.array<?xf32>>>) map_clauses(tofrom) capture(ByRef) name("x") -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
// CHECK:           %[[Y_MAP:.*]] = omp.map.info var_ptr(%[[Y]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>, !fir.box<!fir.heap<!fir.array<?xf32>>>) map_clauses(tofrom) capture(ByRef) name("y") -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
// CHECK:           %[[X_STORAGE:.*]] = omp.map.info var_ptr(%[[X]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>, !fir.box<!fir.heap<!fir.array<?xf32>>>) map_clauses(storage) capture(ByRef) name("x") -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
// CHECK:           %[[Y_STORAGE:.*]] = omp.map.info var_ptr(%[[Y]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>, !fir.box<!fir.heap<!fir.array<?xf32>>>) map_clauses(storage) capture(ByRef) name("y") -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
// CHECK:           omp.target_data map_entries(%[[X_MAP]], %[[Y_MAP]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) {
// CHECK:             %[[CACHE:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xf32>>>
// CHECK:             %[[CACHE_FROM:.*]] = omp.map.info var_ptr(%[[CACHE]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>, !fir.box<!fir.heap<!fir.array<?xf32>>>) map_clauses(from) capture(ByRef) name("__flang_workdistribute_from") -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
// CHECK:             %[[CACHE_TO:.*]] = omp.map.info var_ptr(%[[CACHE]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>, !fir.box<!fir.heap<!fir.array<?xf32>>>) map_clauses(to) capture(ByRef) name("__flang_workdistribute_to") -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
// CHECK:             %[[MLIR_0:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:             %[[CONSTANT_0:.*]] = arith.constant 0 : index
// CHECK:             %[[CONSTANT_1:.*]] = arith.constant 1 : index
// CHECK:             %[[CONSTANT_2:.*]] = arith.constant 9 : index
// CHECK:             %[[CONSTANT_3:.*]] = arith.constant 5.000000e+00 : f32
// CHECK:             %[[OLD_BOX:.*]] = fir.load %[[X]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
// CHECK:             %[[OTHER:.*]] = fir.load %[[Y]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
// CHECK:             fir.store %[[OTHER]] to %[[X]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
// CHECK:             fir.store %[[OLD_BOX]] to %[[CACHE]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
// CHECK:             omp.target kernel_type(spmd) map_entries(%[[X_STORAGE]] -> %[[X_ARG:.*]], %[[Y_STORAGE]] -> %[[Y_ARG:.*]], %[[CACHE_TO]] -> %[[CACHE_ARG:.*]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) {
// CHECK:               %[[BOX:.*]] = fir.load %[[CACHE_ARG]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
// CHECK:               %[[CONSTANT_4:.*]] = arith.constant 0 : index
// CHECK:               %[[CONSTANT_5:.*]] = arith.constant 1 : index
// CHECK:               %[[CONSTANT_6:.*]] = arith.constant 9 : index
// CHECK:               %[[CONSTANT_7:.*]] = arith.constant 5.000000e+00 : f32
// CHECK:               omp.teams {
// CHECK:                 omp.parallel {
// CHECK:                   omp.distribute {
// CHECK:                     omp.wsloop {
// CHECK:                       omp.loop_nest (%[[IV:.*]]) : index = (%[[CONSTANT_4]]) to (%[[CONSTANT_6]]) inclusive step (%[[CONSTANT_5]]) {
// CHECK:                         %[[BOX_ADDR_0:.*]] = fir.box_addr %[[BOX]] : (!fir.box<!fir.heap<!fir.array<?xf32>>>) -> !fir.heap<!fir.array<?xf32>>
// CHECK:                         %[[COORDINATE_OF_0:.*]] = fir.coordinate_of %[[BOX_ADDR_0]], %[[IV]] : (!fir.heap<!fir.array<?xf32>>, index) -> !fir.ref<f32>
// CHECK:                         fir.store %[[CONSTANT_7]] to %[[COORDINATE_OF_0]] : !fir.ref<f32>
// CHECK:                         omp.yield
// CHECK:                       }
// CHECK:                     } {omp.composite}
// CHECK:                   } {omp.composite}
// CHECK:                   omp.terminator
// CHECK:                 } {omp.composite}
// CHECK:                 omp.terminator
// CHECK:               } {omp.combined}
// CHECK:               omp.terminator
// CHECK:             } {omp.combined}
// CHECK:             omp.terminator
// CHECK:           }
// CHECK:           return
// CHECK:         }
func.func @cache_written_descriptor(%arg0: !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>, %arg1: !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) {
  %map0 = omp.map.info var_ptr(%arg0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>, !fir.box<!fir.heap<!fir.array<?xf32>>>) map_clauses(tofrom) capture(ByRef) name("x") -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  %map1 = omp.map.info var_ptr(%arg1 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>, !fir.box<!fir.heap<!fir.array<?xf32>>>) map_clauses(tofrom) capture(ByRef) name("y") -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  omp.target kernel_type(generic) map_entries(%map0 -> %bx, %map1 -> %by : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c9 = arith.constant 9 : index
    %cst = arith.constant 5.000000e+00 : f32
    %box = fir.load %bx : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
    %other = fir.load %by : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
    fir.store %other to %bx : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
    omp.teams {
      omp.workdistribute {
        fir.do_loop %iv = %c0 to %c9 step %c1 unordered {
          %addr = fir.box_addr %box : (!fir.box<!fir.heap<!fir.array<?xf32>>>) -> !fir.heap<!fir.array<?xf32>>
          %coor = fir.coordinate_of %addr, %iv : (!fir.heap<!fir.array<?xf32>>, index) -> !fir.ref<f32>
          fir.store %cst to %coor : !fir.ref<f32>
        }
        omp.terminator
      }
      omp.terminator
    }
    omp.terminator
  }
  return
}
}

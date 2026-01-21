// RUN: not fir-opt --omp-function-filtering -o - %s 2>&1 | FileCheck %s

module attributes {omp.is_gpu = true, omp.is_target_device = true} {
  // CHECK: not yet implemented: Reduction of dynamically-shaped arrays are not supported yet on the GPU.
  omp.declare_reduction @add_reduction_byref_box_heap_Uxi32 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>> attributes {byref_element_type = !fir.array<?xi32>} alloc {
    %0 = fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>>
    omp.yield(%0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>)
  } init {
  ^bb0(%arg0: !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, %arg1: !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>):
    omp.yield(%arg1 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>)
  } combiner {
  ^bb0(%arg0: !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, %arg1: !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>):
    omp.yield(%arg0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>)
  }

  func.func @foo(%ia : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) {
    %ia.map = omp.map.info var_ptr(%ia : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.box<!fir.heap<!fir.array<?xi32>>>) map_clauses(always, implicit, to) capture(ByRef) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>> {name = "ia"}

    omp.target map_entries(%ia.map -> %arg0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) {
      omp.parallel {
        %c1_i32 = arith.constant 1 : i32
        omp.wsloop reduction(byref @add_reduction_byref_box_heap_Uxi32 %arg0 -> %arg1 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) {
          omp.loop_nest (%arg2) : i32 = (%c1_i32) to (%c1_i32) inclusive step (%c1_i32) {
            omp.yield
          }
        }
        omp.terminator
      }
      omp.terminator
    }
    return
  }
}

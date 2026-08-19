// RUN: fir-opt --omp-unimplemented-device-check -verify-diagnostics %s

module attributes {omp.is_gpu = true, omp.is_target_device = true} {
  // expected-error @below {{not yet implemented: Reduction of dynamically-shaped arrays on the GPU.}}
  omp.declare_reduction @red1 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>> attributes {byref_element_type = !fir.array<?xi32>} alloc {
    %0 = fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>>
    omp.yield(%0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>)
  } init {
  ^bb0(%arg0: !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, %arg1: !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>):
    omp.yield(%arg1 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>)
  } combiner {
  ^bb0(%arg0: !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, %arg1: !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>):
    omp.yield(%arg0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>)
  }

  func.func @f1(%ia : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) {
    %ia.map = omp.map.info var_ptr(%ia : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.box<!fir.heap<!fir.array<?xi32>>>) map_clauses(always, implicit, to) capture(ByRef) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>> {name = "ia"}

    omp.target kernel_type(spmd) map_entries(%ia.map -> %arg0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) {
      omp.parallel {
        %c1_i32 = arith.constant 1 : i32
        omp.wsloop reduction(byref @red1 %arg0 -> %arg1 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) {
          omp.loop_nest (%arg2) : i32 = (%c1_i32) to (%c1_i32) inclusive step (%c1_i32) {
            omp.yield
          }
        }
        omp.terminator
      } {omp.combined}
      omp.terminator
    } {omp.combined}

    return
  }

  // expected-error @below {{not yet implemented: Reduction of dynamically-shaped arrays on the GPU.}}
  omp.declare_reduction @red2 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>> attributes {byref_element_type = !fir.array<?xi32>} alloc {
    %0 = fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>>
    omp.yield(%0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>)
  } init {
  ^bb0(%arg0: !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, %arg1: !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>):
    omp.yield(%arg1 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>)
  } combiner {
  ^bb0(%arg0: !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, %arg1: !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>):
    omp.yield(%arg0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>)
  }

  // expected-error @below {{not yet implemented: Reduction of dynamically-shaped arrays on the GPU.}}
  omp.declare_reduction @red3 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>> attributes {byref_element_type = !fir.array<?xi32>} alloc {
    %0 = fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>>
    omp.yield(%0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>)
  } init {
  ^bb0(%arg0: !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, %arg1: !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>):
    omp.yield(%arg1 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>)
  } combiner {
  ^bb0(%arg0: !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, %arg1: !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>):
    omp.yield(%arg0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>)
  }

  func.func @f2(%ia : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) attributes {omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (enter), automap = false>} {
    %c1_i32 = arith.constant 1 : i32
    omp.wsloop reduction(byref @red2 %ia -> %arg0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) {
      omp.loop_nest (%arg1) : i32 = (%c1_i32) to (%c1_i32) inclusive step (%c1_i32) {
        omp.yield
      }
    }
    omp.wsloop reduction(byref @red3 %ia -> %arg0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) {
      omp.loop_nest (%arg1) : i32 = (%c1_i32) to (%c1_i32) inclusive step (%c1_i32) {
        omp.yield
      }
    }
    return
  }

  // This emits no errors, as it's not accessed from target device code.
  omp.declare_reduction @red4 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>> attributes {byref_element_type = !fir.array<?xi32>} alloc {
    %0 = fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>>
    omp.yield(%0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>)
  } init {
  ^bb0(%arg0: !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, %arg1: !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>):
    omp.yield(%arg1 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>)
  } combiner {
  ^bb0(%arg0: !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, %arg1: !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>):
    omp.yield(%arg0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>)
  }

  func.func @f3(%ia : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) {
    %c1_i32 = arith.constant 1 : i32
    omp.wsloop reduction(byref @red4 %ia -> %arg0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) {
      omp.loop_nest (%arg1) : i32 = (%c1_i32) to (%c1_i32) inclusive step (%c1_i32) {
        omp.yield
      }
    }
    return
  }
}

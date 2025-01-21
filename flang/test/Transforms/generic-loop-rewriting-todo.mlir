// RUN: fir-opt --omp-generic-loop-conversion -verify-diagnostics %s

func.func @_QPtarget_loop() {
  %c0 = arith.constant 0 : i32
  %c10 = arith.constant 10 : i32
  %c1 = arith.constant 1 : i32
  // expected-error@below {{not yet implemented: Standalone `omp loop` directive}}
  omp.loop {
    omp.loop_nest (%arg3) : i32 = (%c0) to (%c10) inclusive step (%c1) {
      omp.yield
    }
  }
  return
}

func.func @_QPtarget_parallel_loop() {
  omp.target {
    omp.parallel {
      %c0 = arith.constant 0 : i32
      %c10 = arith.constant 10 : i32
      %c1 = arith.constant 1 : i32
      // expected-error@below {{not yet implemented: Combined `omp target parallel loop` directive}}
      omp.loop {
        omp.loop_nest (%arg3) : i32 = (%c0) to (%c10) inclusive step (%c1) {
          omp.yield
        }
      }
      omp.terminator
    }
    omp.terminator
  }
  return
}

func.func @_QPtarget_loop_bind() {
  omp.target {
    omp.teams {
      %c0 = arith.constant 0 : i32
      %c10 = arith.constant 10 : i32
      %c1 = arith.constant 1 : i32
      // expected-error@below {{not yet implemented: Unhandled clause bind in omp.loop operation}}
      omp.loop bind(thread) {
        omp.loop_nest (%arg3) : i32 = (%c0) to (%c10) inclusive step (%c1) {
          omp.yield
        }
      }
      omp.terminator
    }
    omp.terminator
  }
  return
}

omp.declare_reduction @add_reduction_i32 : i32 init {
  ^bb0(%arg0: i32):
    %c0_i32 = arith.constant 0 : i32
    omp.yield(%c0_i32 : i32)
  } combiner {
  ^bb0(%arg0: i32, %arg1: i32):
    %0 = arith.addi %arg0, %arg1 : i32
    omp.yield(%0 : i32)
  }

func.func @_QPtarget_loop_order() {

  omp.target {
    omp.teams {
      %c0 = arith.constant 0 : i32
      %c10 = arith.constant 10 : i32
      %c1 = arith.constant 1 : i32
      %sum = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFtest_orderEi"}

      // expected-error@below {{not yet implemented: Unhandled clause reduction in omp.loop operation}}
      omp.loop reduction(@add_reduction_i32 %sum -> %arg2 : !fir.ref<i32>) {
        omp.loop_nest (%arg3) : i32 = (%c0) to (%c10) inclusive step (%c1) {
          omp.yield
        }
      }
      omp.terminator
    }
    omp.terminator
  }
  return
}

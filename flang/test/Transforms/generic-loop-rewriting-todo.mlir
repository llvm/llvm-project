// RUN: fir-opt --omp-generic-loop-conversion -verify-diagnostics %s
func.func @_QPloop_order() {
  omp.teams {
    %c0 = arith.constant 0 : i32
    %c10 = arith.constant 10 : i32
    %c1 = arith.constant 1 : i32

    // expected-error@below {{not yet implemented: Unhandled clause order in omp.loop operation}}
    omp.loop order(reproducible:concurrent) {
      omp.loop_nest (%arg3) : i32 = (%c0) to (%c10) inclusive step (%c1) {
        omp.yield
      }
    }
    omp.terminator
  }
  return
}

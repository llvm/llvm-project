// RUN: not fir-opt --lower-workshare --allow-unregistered-dialect %s 2>&1 | FileCheck %s

// Broadcasting a reference-typed value across threads would require a
// !fir.ref<!fir.ref<...>> slot, which FIR forbids (see fir::ReferenceType::verify).
// Rather than fall back to a !fir.ref<!fir.ptr<...>> workaround, the pass emits a
// TODO. In practice no Fortran lowering reaches this branch: every op that yields
// a !fir.ref is pure and is therefore parallelized directly rather than being
// reloaded here. This synthetic test uses an unregistered op with unknown memory
// effects to force a reference-typed value to be broadcast.

// CHECK: not yet implemented: unsupported value in OpenMP workshare region: a reference used across the region cannot be made available to all threads

func.func @wsfunc(%arg0: !fir.ref<!fir.array<10xi32>>) {
  omp.parallel {
    omp.workshare {
      %c1 = arith.constant 1 : index
      %c10 = arith.constant 10 : index
      // Impure (unregistered) op producing a reference value that is used inside
      // the workshare loop, i.e. live outside the omp.single region.
      %addr = "test.get_addr"(%arg0)
              : (!fir.ref<!fir.array<10xi32>>) -> !fir.ref<i32>
      omp.workshare.loop_wrapper {
        omp.loop_nest (%iv) : index = (%c1) to (%c10) inclusive step (%c1) {
          %v = fir.load %addr : !fir.ref<i32>
          %e = fir.coordinate_of %arg0, %iv
               : (!fir.ref<!fir.array<10xi32>>, index) -> !fir.ref<i32>
          fir.store %v to %e : !fir.ref<i32>
          omp.yield
        }
      }
      omp.terminator
    }
    omp.terminator
  }
  return
}

// RUN: fir-opt --lower-workshare --allow-unregistered-dialect %s | FileCheck %s

// Check that the safe to parallelize `fir.declare` op will not be parallelized
// due to its operand %alloc not being reloaded outside the omp.single.

func.func @foo() {
  %c0 = arith.constant 0 : index
  omp.workshare {
    %alloc = fir.allocmem !fir.array<?xf32>, %c0 {bindc_name = ".tmp.forall", uniq_name = ""}
    %shape = fir.shape %c0 : (index) -> !fir.shape<1>
    %declare = fir.declare %alloc(%shape) {uniq_name = ".tmp.forall"} : (!fir.heap<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.heap<!fir.array<?xf32>>
    fir.freemem %alloc : !fir.heap<!fir.array<?xf32>>
    omp.terminator
  }
  return
}

// CHECK:    omp.single nowait
// CHECK:      fir.allocmem
// CHECK:      fir.shape
// CHECK:      fir.declare
// CHECK:      fir.freemem
// CHECK:      omp.terminator
// CHECK:    }
// CHECK:    omp.barrier

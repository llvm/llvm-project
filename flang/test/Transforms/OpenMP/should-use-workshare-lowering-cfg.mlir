// RUN: fir-opt --bufferize-hlfir %s 2>&1 | FileCheck %s

// CHECK: warning: omp workshare with unstructured control flow currently unsupported.
func.func @warn_cfg(%arg: !fir.ref<!fir.array<42xi32>>, %idx : index) {
  omp.workshare {
  ^bb1:
    %c42 = arith.constant 42 : index
    %c1_i32 = arith.constant 1 : i32
    %shape = fir.shape %c42 : (index) -> !fir.shape<1>
    %array:2 = hlfir.declare %arg(%shape) {uniq_name = "array"} : (!fir.ref<!fir.array<42xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<42xi32>>, !fir.ref<!fir.array<42xi32>>)
    %elemental = hlfir.elemental %shape unordered : (!fir.shape<1>) -> !hlfir.expr<42xi32> {
    ^bb0(%i: index):
      hlfir.yield_element %c1_i32 : i32
    }
    hlfir.assign %elemental to %array#0 : !hlfir.expr<42xi32>, !fir.ref<!fir.array<42xi32>>
    hlfir.destroy %elemental : !hlfir.expr<42xi32>
    cf.br ^bb2
  ^bb2:
    omp.terminator
  }
  return
}

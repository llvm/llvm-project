// RUN: fir-opt %s --opt-bufferization | FileCheck %s

// Test that alias analysis doesn't crash determining if the arguments to
// hlfir.assign alias.
// CHECK: omp.private {type = firstprivate} @_QFFbEl_firstprivate_box_Uxi32

// TODO: we can't currently optimize this assign because alias analysis doesn't
// know that the block arguments of the copy region cannot alias.

omp.private {type = firstprivate} @_QFFbEl_firstprivate_box_Uxi32 : !fir.box<!fir.array<?xi32>> init {
^bb0(%arg0: !fir.ref<!fir.box<!fir.array<?xi32>>>, %arg1: !fir.ref<!fir.box<!fir.array<?xi32>>>):
  %0 = fir.load %arg0 : !fir.ref<!fir.box<!fir.array<?xi32>>>
  %c0 = arith.constant 0 : index
  %1:3 = fir.box_dims %0, %c0 : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
  %2 = fir.shape %1#1 : (index) -> !fir.shape<1>
  %3 = fir.allocmem !fir.array<?xi32>, %1#1 {bindc_name = ".tmp", uniq_name = ""}
  %true = arith.constant true
  %4:2 = hlfir.declare %3(%2) {uniq_name = ".tmp"} : (!fir.heap<!fir.array<?xi32>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xi32>>, !fir.heap<!fir.array<?xi32>>)
  %c0_0 = arith.constant 0 : index
  %5:3 = fir.box_dims %0, %c0_0 : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
  %6 = fir.shape_shift %5#0, %5#1 : (index, index) -> !fir.shapeshift<1>
  %7 = fir.rebox %4#0(%6) : (!fir.box<!fir.array<?xi32>>, !fir.shapeshift<1>) -> !fir.box<!fir.array<?xi32>>
  fir.store %7 to %arg1 : !fir.ref<!fir.box<!fir.array<?xi32>>>
  omp.yield(%arg1 : !fir.ref<!fir.box<!fir.array<?xi32>>>)
} copy {
^bb0(%arg0: !fir.ref<!fir.box<!fir.array<?xi32>>>, %arg1 : !fir.ref<!fir.box<!fir.array<?xi32>>>):
  %0 = fir.load %arg0 {test.ptr = "load_from_block_arg"} : !fir.ref<!fir.box<!fir.array<?xi32>>>
  hlfir.assign %0 to %arg1 : !fir.box<!fir.array<?xi32>>, !fir.ref<!fir.box<!fir.array<?xi32>>>
  omp.yield(%arg1 : !fir.ref<!fir.box<!fir.array<?xi32>>>)
} dealloc {
^bb0(%arg0: !fir.ref<!fir.box<!fir.array<?xi32>>>):
  %0 = fir.load %arg0 : !fir.ref<!fir.box<!fir.array<?xi32>>>
  %1 = fir.box_addr %0 : (!fir.box<!fir.array<?xi32>>) -> !fir.ref<!fir.array<?xi32>>
  %2 = fir.convert %1 : (!fir.ref<!fir.array<?xi32>>) -> i64
  %c0_i64 = arith.constant 0 : i64
  %3 = arith.cmpi ne, %2, %c0_i64 : i64
  fir.if %3 {
    %4 = fir.convert %1 : (!fir.ref<!fir.array<?xi32>>) -> !fir.heap<!fir.array<?xi32>>
    fir.freemem %4 : !fir.heap<!fir.array<?xi32>>
  }
  omp.yield
}

// Test that InlineHLFIRAssign successfully inlines hlfir.assign operations
// in omp.private copy regions when the fix for omp.private block argument
// aliasing is present. Without the fix, alias analysis returns MayAlias and
// InlineHLFIRAssign fails.
//
// RUN: fir-opt %s --inline-hlfir-assign --split-input-file | FileCheck %s

// Fortran source (simplified):
//   program minimal
//     integer :: arr(8)
//     arr(1) = 1
//     !$omp target firstprivate(arr)
//     arr(1) = 42
//     !$omp end target
//   end program

omp.private {type = firstprivate} @arr_privatizer : !fir.box<!fir.array<8xi32>> init {
^bb0(%arg0: !fir.ref<!fir.box<!fir.array<8xi32>>>, %arg1: !fir.ref<!fir.box<!fir.array<8xi32>>>):
  %c8 = arith.constant 8 : index
  %c1 = arith.constant 1 : index
  %0 = fir.shape %c8 : (index) -> !fir.shape<1>
  %1 = fir.allocmem !fir.array<8xi32> {bindc_name = ".tmp", uniq_name = ""}
  %2:2 = hlfir.declare %1(%0) {uniq_name = ".tmp"} : (!fir.heap<!fir.array<8xi32>>, !fir.shape<1>) -> (!fir.heap<!fir.array<8xi32>>, !fir.heap<!fir.array<8xi32>>)
  %3 = fir.shape_shift %c1, %c8 : (index, index) -> !fir.shapeshift<1>
  %4 = fir.embox %2#0(%3) : (!fir.heap<!fir.array<8xi32>>, !fir.shapeshift<1>) -> !fir.box<!fir.array<8xi32>>
  fir.store %4 to %arg1 : !fir.ref<!fir.box<!fir.array<8xi32>>>
  omp.yield(%arg1 : !fir.ref<!fir.box<!fir.array<8xi32>>>)
} copy {
^bb0(%arg0: !fir.ref<!fir.box<!fir.array<8xi32>>>, %arg1: !fir.ref<!fir.box<!fir.array<8xi32>>>):
  // This hlfir.assign should be inlined into a loop by InlineHLFIRAssign
  // because alias analysis should return NoAlias for %0 (loaded from %arg0 mold)
  // vs %arg1 (private copy).
  %0 = fir.load %arg0 : !fir.ref<!fir.box<!fir.array<8xi32>>>
  hlfir.assign %0 to %arg1 : !fir.box<!fir.array<8xi32>>, !fir.ref<!fir.box<!fir.array<8xi32>>>
  omp.yield(%arg1 : !fir.ref<!fir.box<!fir.array<8xi32>>>)
}

// CHECK-LABEL: omp.private {type = firstprivate} @arr_privatizer
// CHECK: } copy {
// CHECK: ^bb0(%[[MOLD:.*]]: !fir.ref<!fir.box<!fir.array<8xi32>>>, %[[PRIV:.*]]: !fir.ref<!fir.box<!fir.array<8xi32>>>):
// CHECK-NOT: hlfir.assign {{.*}} to {{.*}} : !fir.box<!fir.array<8xi32>>
// CHECK: fir.do_loop
// CHECK: hlfir.designate
// CHECK: hlfir.designate
// CHECK: hlfir.assign {{.*}} to {{.*}} : i32
// CHECK: omp.yield

func.func @test_omp_private_copy_inlined() {
  return
}

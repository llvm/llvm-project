// RUN: fir-opt -flang-licm --split-input-file %s | FileCheck %s

// A call that is passed the address of the descriptor of an allocatable may
// modify the data of the allocatable, so a read of this data is not hoisted
// out of a loop that contains the call. The loops have a static trip count,
// so the reads are hoisted when nothing in the loop modifies the data.

// subroutine test(s)
//   integer :: s
//   integer, allocatable :: a
//   associate (y => a)
//     do i = 1, 3
//       call bump(a)
//       s = s + y
//     end do
//   end associate
// end subroutine
//
// The allocatable dummy of bump has no intent.

// CHECK-LABEL:   func.func @_QPtest(
// CHECK:           %[[Y:.*]] = fir.declare %{{.*}} uniq_name("_QFtestEy")
// CHECK:           scf.for
// CHECK-NEXT:        fir.call @_QPbump(
// CHECK-NEXT:        fir.load
// CHECK-NEXT:        fir.load %[[Y]] : !fir.heap<i32>
func.func @_QPtest(%arg0: !fir.ref<i32> {fir.bindc_name = "s"}) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.alloca !fir.box<!fir.heap<i32>> {bindc_name = "a", uniq_name = "_QFtestEa"}
  %2 = fir.declare %1 uniq_name("_QFtestEa") fortran_attrs<allocatable> : (!fir.ref<!fir.box<!fir.heap<i32>>>) -> !fir.ref<!fir.box<!fir.heap<i32>>>
  %3 = fir.declare %arg0 dummy_scope %0 arg 1 uniq_name("_QFtestEs") : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
  %4 = fir.load %2 : !fir.ref<!fir.box<!fir.heap<i32>>>
  %5 = fir.box_addr %4 : (!fir.box<!fir.heap<i32>>) -> !fir.heap<i32>
  %6 = fir.declare %5 uniq_name("_QFtestEy") : (!fir.heap<i32>) -> !fir.heap<i32>
  scf.for %arg1 = %c0 to %c3 step %c1 {
    fir.call @_QPbump(%2) : (!fir.ref<!fir.box<!fir.heap<i32>>>) -> ()
    %7 = fir.load %3 : !fir.ref<i32>
    %8 = fir.load %6 : !fir.heap<i32>
    %9 = arith.addi %7, %8 : i32
    fir.store %9 to %3 : !fir.ref<i32>
  }
  return
}
func.func private @_QPbump(!fir.ref<!fir.box<!fir.heap<i32>>>)

// -----

// The same with an intent(in) allocatable dummy: the call only reads the data
// of a, so the read of y is hoisted.

// CHECK-LABEL:   func.func @_QPtest_in(
// CHECK:           %[[Y:.*]] = fir.declare %{{.*}} uniq_name("_QFtest_inEy")
// CHECK:           fir.load %[[Y]] : !fir.heap<i32>
// CHECK:           scf.for
// CHECK-NEXT:        fir.call @_QPpeek(
// CHECK-NEXT:        fir.load
// CHECK-NEXT:        arith.addi
func.func @_QPtest_in(%arg0: !fir.ref<i32> {fir.bindc_name = "s"}) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.alloca !fir.box<!fir.heap<i32>> {bindc_name = "a", uniq_name = "_QFtest_inEa"}
  %2 = fir.declare %1 uniq_name("_QFtest_inEa") fortran_attrs<allocatable> : (!fir.ref<!fir.box<!fir.heap<i32>>>) -> !fir.ref<!fir.box<!fir.heap<i32>>>
  %3 = fir.declare %arg0 dummy_scope %0 arg 1 uniq_name("_QFtest_inEs") : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
  %4 = fir.load %2 : !fir.ref<!fir.box<!fir.heap<i32>>>
  %5 = fir.box_addr %4 : (!fir.box<!fir.heap<i32>>) -> !fir.heap<i32>
  %6 = fir.declare %5 uniq_name("_QFtest_inEy") : (!fir.heap<i32>) -> !fir.heap<i32>
  scf.for %arg1 = %c0 to %c3 step %c1 {
    fir.call @_QPpeek(%2) : (!fir.ref<!fir.box<!fir.heap<i32>>>) -> ()
    %7 = fir.load %3 : !fir.ref<i32>
    %8 = fir.load %6 : !fir.heap<i32>
    %9 = arith.addi %7, %8 : i32
    fir.store %9 to %3 : !fir.ref<i32>
  }
  return
}
func.func private @_QPpeek(!fir.ref<!fir.box<!fir.heap<i32>>> {fir.fortran_attrs = #fir.var_attrs<intent_in>})

! Test assumed-rank dummy argument that is not present in
! all ENTRY statements.
! RUN: bbc -emit-hlfir -allow-assumed-rank -o - %s | FileCheck %s

subroutine test_main_entry(x)
  real :: x(..)
  interface
  subroutine some_proc(x)
    real :: x(..)
  end subroutine
  end interface
  call some_proc(x)
entry test_alternate_entry()
  call the_end()
end subroutine
! CHECK-LABEL:   func.func @_QPtest_main_entry(
! CHECK-SAME:                                  %[[VAL_0:.*]]: !fir.box<!fir.array<*:f32>> {fir.bindc_name = "x"}) {
! CHECK:           %[[VAL_1:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %[[VAL_1]] {uniq_name = "_QFtest_main_entryEx"} : (!fir.box<!fir.array<*:f32>>, !fir.dscope) -> (!fir.box<!fir.array<*:f32>>, !fir.box<!fir.array<*:f32>>)

! CHECK-LABEL:   func.func @_QPtest_alternate_entry() {
! CHECK:           %[[VAL_0:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<*:f32>>>
! CHECK:           %[[VAL_1:.*]] = fir.zero_bits !fir.heap<f32>
! CHECK:           %[[VAL_2:.*]] = fir.embox %[[VAL_1]] : (!fir.heap<f32>) -> !fir.box<!fir.heap<f32>>
! CHECK:           %[[VAL_3:.*]] = fir.convert %[[VAL_2]] : (!fir.box<!fir.heap<f32>>) -> !fir.box<!fir.heap<!fir.array<*:f32>>>
! CHECK:           fir.store %[[VAL_3]] to %[[VAL_0]] : !fir.ref<!fir.box<!fir.heap<!fir.array<*:f32>>>>
! CHECK:           %[[VAL_4:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.heap<!fir.array<*:f32>>>>
! CHECK:           %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_4]] {uniq_name = "_QFtest_main_entryEx"} : (!fir.box<!fir.heap<!fir.array<*:f32>>>) -> (!fir.box<!fir.heap<!fir.array<*:f32>>>, !fir.box<!fir.heap<!fir.array<*:f32>>>)

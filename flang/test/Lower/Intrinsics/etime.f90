! RUN: bbc -emit-fir %s -o - | FileCheck %s

! CHECK-LABEL: func.func @_QPetime_test(
! CHECK-SAME: %[[valuesArg:.*]]: !fir.ref<!fir.array<2xf32>> {fir.bindc_name = "values"},
! CHECK-SAME: %[[timeArg:.*]]: !fir.ref<f32> {fir.bindc_name = "time"}) {
subroutine etime_test(values, time)
  REAL(4), DIMENSION(2) :: values
  REAL(4) :: time
  call etime(values, time)
  ! CHECK-NEXT:        %[[c9:.*]] = arith.constant 9 : i32
  ! CHECK-NEXT:        %[[c2:.*]] = arith.constant 2 : index
  ! CHECK-NEXT:        %[[DSCOPE:.*]] = fir.dummy_scope : !fir.dscope
  ! CHECK-NEXT:        %[[timeDeclare:.*]] = fir.declare %[[timeArg]] dummy_scope %[[DSCOPE]] {uniq_name = "_QFetime_testEtime"} : (!fir.ref<f32>, !fir.dscope) -> !fir.ref<f32>
  ! CHECK-NEXT:        %[[shape:.*]] = fir.shape %[[c2]] : (index) -> !fir.shape<1>
  ! CHECK-NEXT:        %[[valuesDeclare:.*]] = fir.declare %[[valuesArg]](%[[shape]]) dummy_scope %[[DSCOPE]] {uniq_name = "_QFetime_testEvalues"} : (!fir.ref<!fir.array<2xf32>>, !fir.shape<1>, !fir.dscope) -> !fir.ref<!fir.array<2xf32>>
  ! CHECK-NEXT:        %[[valuesBox:.*]] = fir.embox %[[valuesDeclare]](%[[shape]]) : (!fir.ref<!fir.array<2xf32>>, !fir.shape<1>) -> !fir.box<!fir.array<2xf32>>
  ! CHECK-NEXT:        %[[timeBox:.*]] = fir.embox %[[timeDeclare]] : (!fir.ref<f32>) -> !fir.box<f32>
  ! CHECK:             %[[values:.*]] = fir.convert %[[valuesBox]] : (!fir.box<!fir.array<2xf32>>) -> !fir.box<none>
  ! CHECK:             %[[time:.*]] = fir.convert %[[timeBox]] : (!fir.box<f32>) -> !fir.box<none>
  ! CHECK:             %[[VAL_9:.*]] = fir.call @_FortranAEtime(%[[values]], %[[time]], %[[VAL_7:.*]], %[[c9]]) fastmath<contract> : (!fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> none
  ! CHECK-NEXT:        return
end subroutine etime_test
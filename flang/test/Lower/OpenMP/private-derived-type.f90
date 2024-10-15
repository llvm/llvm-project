! RUN: %flang_fc1 -emit-hlfir -fopenmp -o - %s | FileCheck %s
! RUN: bbc -emit-hlfir -fopenmp -o - %s | FileCheck %s

subroutine s4
  type y3
    integer,allocatable::x
  end type y3
  type(y3)::v
  !$omp parallel
  !$omp do private(v)
  do i=1,10
    v%x=1
  end do
  !$omp end do
  !$omp end parallel
end subroutine s4


! CHECK-LABEL:   func.func @_QPs4() {
!                  Example of how the lowering for regular derived type variables:
! CHECK:           %[[VAL_8:.*]] = fir.alloca !fir.type<_QFs4Ty3{x:!fir.box<!fir.heap<i32>>}> {bindc_name = "v", uniq_name = "_QFs4Ev"}
! CHECK:           %[[VAL_9:.*]]:2 = hlfir.declare %[[VAL_8]] {uniq_name = "_QFs4Ev"} : (!fir.ref<!fir.type<_QFs4Ty3{x:!fir.box<!fir.heap<i32>>}>>) -> (!fir.ref<!fir.type<_QFs4Ty3{x:!fir.box<!fir.heap<i32>>}>>, !fir.ref<!fir.type<_QFs4Ty3{x:!fir.box<!fir.heap<i32>>}>>)
! CHECK:           %[[VAL_10:.*]] = fir.embox %[[VAL_9]]#1 : (!fir.ref<!fir.type<_QFs4Ty3{x:!fir.box<!fir.heap<i32>>}>>) -> !fir.box<!fir.type<_QFs4Ty3{x:!fir.box<!fir.heap<i32>>}>>
! CHECK:           %[[VAL_11:.*]] = fir.address_of
! CHECK:           %[[VAL_12:.*]] = arith.constant 4 : i32
! CHECK:           %[[VAL_13:.*]] = fir.convert %[[VAL_10]] : (!fir.box<!fir.type<_QFs4Ty3{x:!fir.box<!fir.heap<i32>>}>>) -> !fir.box<none>
! CHECK:           %[[VAL_14:.*]] = fir.convert %[[VAL_11]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
! CHECK:           %[[VAL_15:.*]] = fir.call @_FortranAInitialize(%[[VAL_13]], %[[VAL_14]], %[[VAL_12]]) fastmath<contract> : (!fir.box<none>, !fir.ref<i8>, i32) -> none
! CHECK:           omp.parallel {
! CHECK:             %[[VAL_23:.*]] = fir.alloca !fir.type<_QFs4Ty3{x:!fir.box<!fir.heap<i32>>}> {bindc_name = "v", pinned, uniq_name = "_QFs4Ev"}
! CHECK:             %[[VAL_24:.*]]:2 = hlfir.declare %[[VAL_23]] {uniq_name = "_QFs4Ev"} : (!fir.ref<!fir.type<_QFs4Ty3{x:!fir.box<!fir.heap<i32>>}>>) -> (!fir.ref<!fir.type<_QFs4Ty3{x:!fir.box<!fir.heap<i32>>}>>, !fir.ref<!fir.type<_QFs4Ty3{x:!fir.box<!fir.heap<i32>>}>>)
! CHECK:             %[[VAL_25:.*]] = fir.embox %[[VAL_24]]#1 : (!fir.ref<!fir.type<_QFs4Ty3{x:!fir.box<!fir.heap<i32>>}>>) -> !fir.box<!fir.type<_QFs4Ty3{x:!fir.box<!fir.heap<i32>>}>>
! CHECK:             %[[VAL_26:.*]] = fir.address_of
! CHECK:             %[[VAL_27:.*]] = arith.constant 4 : i32
! CHECK:             %[[VAL_28:.*]] = fir.convert %[[VAL_25]] : (!fir.box<!fir.type<_QFs4Ty3{x:!fir.box<!fir.heap<i32>>}>>) -> !fir.box<none>
! CHECK:             %[[VAL_29:.*]] = fir.convert %[[VAL_26]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
!                    Check we do call FortranAInitialize on the derived type
! CHECK:             %[[VAL_30:.*]] = fir.call @_FortranAInitialize(%[[VAL_28]], %[[VAL_29]], %[[VAL_27]]) fastmath<contract> : (!fir.box<none>, !fir.ref<i8>, i32) -> none
! CHECK:             omp.wsloop {
! CHECK:           }
! CHECK:           %[[VAL_39:.*]] = fir.embox %[[VAL_9]]#1 : (!fir.ref<!fir.type<_QFs4Ty3{x:!fir.box<!fir.heap<i32>>}>>) -> !fir.box<!fir.type<_QFs4Ty3{x:!fir.box<!fir.heap<i32>>}>>
! CHECK:           %[[VAL_40:.*]] = fir.convert %[[VAL_39]] : (!fir.box<!fir.type<_QFs4Ty3{x:!fir.box<!fir.heap<i32>>}>>) -> !fir.box<none>
!                  Check the derived type is destroyed
! CHECK:           %[[VAL_41:.*]] = fir.call @_FortranADestroy(%[[VAL_40]]) fastmath<contract> : (!fir.box<none>) -> none
! CHECK:           return
! CHECK:         }

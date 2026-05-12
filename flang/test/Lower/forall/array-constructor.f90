! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

subroutine ac1(arr,n)
  integer :: arr(:), n
  forall (i=1:n:2)
     arr(i:i+2) = func((/i/))
  end forall
contains
   pure integer function func(a)
    integer, intent(in) :: a(:)
    func = a(1)
  end function func
end subroutine ac1

! CHECK-LABEL: func.func @_QPac1(
! CHECK-SAME: %[[ARR_ARG:.*]]: !fir.box<!fir.array<?xi32>> {{.*}}, %[[N_ARG:.*]]: !fir.ref<i32> {{.*}}) {
! CHECK: %[[ARR:.*]]:2 = hlfir.declare %[[ARR_ARG]]
! CHECK: %[[N:.*]]:2 = hlfir.declare %[[N_ARG]]
! CHECK: %[[C1_I32:.*]] = arith.constant 1 : i32
! CHECK: %[[N_VAL:.*]] = fir.load %[[N]]#0 : !fir.ref<i32>
! CHECK: %[[C2_I32:.*]] = arith.constant 2 : i32
! CHECK: hlfir.forall lb {
! CHECK:   hlfir.yield %[[C1_I32]] : i32
! CHECK: } ub {
! CHECK:   hlfir.yield %[[N_VAL]] : i32
! CHECK: } step {
! CHECK:   hlfir.yield %[[C2_I32]] : i32
! CHECK: }  (%[[I_VAL:.*]]: i32) {
! CHECK:   %[[I_REF:.*]] = hlfir.forall_index "i" %[[I_VAL]] : (i32) -> !fir.ref<i32>
! CHECK:   hlfir.region_assign {
! CHECK:     %[[C1_IDX:.*]] = arith.constant 1 : index
! CHECK:     %[[TMP:.*]] = fir.allocmem !fir.array<1xi32>
! CHECK:     %[[SHAPE:.*]] = fir.shape %[[C1_IDX]] : (index) -> !fir.shape<1>
! CHECK:     %[[TMP_DECL:.*]]:2 = hlfir.declare %[[TMP]](%[[SHAPE]])
! CHECK:     %[[I_VAL_LOAD:.*]] = fir.load %[[I_REF]] : !fir.ref<i32>
! CHECK:     %[[DSG:.*]] = hlfir.designate %[[TMP_DECL]]#0 ({{.*}})  : (!fir.heap<!fir.array<1xi32>>, index) -> !fir.ref<i32>
! CHECK:     hlfir.assign %[[I_VAL_LOAD]] to %[[DSG]] : i32, !fir.ref<i32>
! CHECK:     %[[EXPR:.*]] = hlfir.as_expr %[[TMP_DECL]]#0 move %{{.*}} : (!fir.heap<!fir.array<1xi32>>, i1) -> !hlfir.expr<1xi32>
! CHECK:     %[[ASSOC:.*]]:3 = hlfir.associate %[[EXPR]](%[[SHAPE]])
! CHECK:     %[[BOX:.*]] = fir.embox %[[ASSOC]]#0(%[[SHAPE]])
! CHECK:     %[[CONV:.*]] = fir.convert %[[BOX]] : (!fir.box<!fir.array<1xi32>>) -> !fir.box<!fir.array<?xi32>>
! CHECK:     %[[RES:.*]] = fir.call @_QFac1Pfunc(%[[CONV]]) {{.*}} : (!fir.box<!fir.array<?xi32>>) -> i32
! CHECK:     hlfir.yield %[[RES]] : i32 cleanup {
! CHECK:       hlfir.end_associate %[[ASSOC]]#1, %[[ASSOC]]#2
! CHECK:       hlfir.destroy %[[EXPR]]
! CHECK:     }
! CHECK:   } to {
! CHECK:     %[[LB:.*]] = fir.convert %{{.*}} : (i64) -> index
! CHECK:     %[[UB:.*]] = fir.convert %{{.*}} : (i64) -> index
! CHECK:     %[[DSG_LHS:.*]] = hlfir.designate %[[ARR]]#0 (%[[LB]]:%[[UB]]:{{.*}})  shape {{.*}} : (!fir.box<!fir.array<?xi32>>, index, index, index, !fir.shape<1>) -> !fir.box<!fir.array<?xi32>>
! CHECK:     hlfir.yield %[[DSG_LHS]] : !fir.box<!fir.array<?xi32>>
! CHECK:   }
! CHECK: }

! CHECK-LABEL: func.func private @_QFac1Pfunc(
! CHECK-SAME: %[[A_ARG:.*]]: !fir.box<!fir.array<?xi32>> {{.*}}) -> i32
! CHECK-SAME: attributes {
! CHECK: %[[A:.*]]:2 = hlfir.declare %[[A_ARG]]
! CHECK: %[[C1:.*]] = arith.constant 1 : index
! CHECK: %[[DSG:.*]] = hlfir.designate %[[A]]#0 (%[[C1]])  : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
! CHECK: %[[VAL:.*]] = fir.load %[[DSG]] : !fir.ref<i32>
! CHECK: hlfir.assign %[[VAL]] to %{{.*}}#0 : i32, !fir.ref<i32>
! CHECK: %[[RET:.*]] = fir.load %{{.*}}#0 : !fir.ref<i32>
! CHECK: return %[[RET]] : i32

subroutine ac2(arr,n)
  integer :: arr(:), n
  forall (i=1:n:2)
     arr(i:i+2) = func((/i/))
  end forall
contains
  pure function func(a)
    integer :: func(3)
    integer, intent(in) :: a(:)
    func = a(1:3)
  end function func
end subroutine ac2

! CHECK-LABEL: func.func @_QPac2(
! CHECK-SAME: %[[ARR_ARG:.*]]: !fir.box<!fir.array<?xi32>> {{.*}}, %[[N_ARG:.*]]: !fir.ref<i32> {{.*}}) {
! CHECK: %[[ARR:.*]]:2 = hlfir.declare %[[ARR_ARG]]
! CHECK: %[[N:.*]]:2 = hlfir.declare %[[N_ARG]]
! CHECK: hlfir.forall lb {
! CHECK:   hlfir.yield %{{.*}} : i32
! CHECK: } ub {
! CHECK:   hlfir.yield %{{.*}} : i32
! CHECK: } step {
! CHECK:   hlfir.yield %{{.*}} : i32
! CHECK: }  (%[[I_VAL:.*]]: i32) {
! CHECK:   %[[I_REF:.*]] = hlfir.forall_index "i" %[[I_VAL]] : (i32) -> !fir.ref<i32>
! CHECK:   hlfir.region_assign {
! CHECK:     %[[EVAL:.*]] = hlfir.eval_in_mem {{.*}} -> !hlfir.expr<3xi32> {
! CHECK:     ^bb0(%[[RES_REF:.*]]: !fir.ref<!fir.array<3xi32>>):
! CHECK:       %[[CALL_RES:.*]] = fir.call @_QFac2Pfunc(%{{.*}}) {{.*}} : (!fir.box<!fir.array<?xi32>>) -> !fir.array<3xi32>
! CHECK:       fir.save_result %[[CALL_RES]] to %[[RES_REF]]({{.*}})
! CHECK:     }
! CHECK:     hlfir.yield %[[EVAL]] : !hlfir.expr<3xi32> cleanup {
! CHECK:       hlfir.destroy %[[EVAL]]
! CHECK:     }
! CHECK:   } to {
! CHECK:     %[[LB:.*]] = fir.convert %{{.*}} : (i64) -> index
! CHECK:     %[[UB:.*]] = fir.convert %{{.*}} : (i64) -> index
! CHECK:     %[[DSG_LHS:.*]] = hlfir.designate %[[ARR]]#0 (%[[LB]]:%[[UB]]:{{.*}})  shape {{.*}} : (!fir.box<!fir.array<?xi32>>, index, index, index, !fir.shape<1>) -> !fir.box<!fir.array<?xi32>>
! CHECK:     hlfir.yield %[[DSG_LHS]] : !fir.box<!fir.array<?xi32>>
! CHECK:   }
! CHECK: }

! CHECK-LABEL: func.func private @_QFac2Pfunc(
! CHECK-SAME: %[[A_ARG:.*]]: !fir.box<!fir.array<?xi32>> {{.*}}) -> !fir.array<3xi32>
! CHECK-SAME: attributes {
! CHECK: %[[A:.*]]:2 = hlfir.declare %[[A_ARG]]
! CHECK: %[[DSG:.*]] = hlfir.designate %[[A]]#0 ({{.*}})  shape %{{.*}} : (!fir.box<!fir.array<?xi32>>, index, index, index, !fir.shape<1>) -> !fir.box<!fir.array<3xi32>>
! CHECK: hlfir.assign %[[DSG]] to %{{.*}}#0 : !fir.box<!fir.array<3xi32>>, !fir.ref<!fir.array<3xi32>>
! CHECK: %[[RES:.*]] = fir.load %{{.*}}#0 : !fir.ref<!fir.array<3xi32>>
! CHECK: return %[[RES]] : !fir.array<3xi32>

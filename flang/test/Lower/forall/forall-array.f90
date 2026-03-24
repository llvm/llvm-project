! Test forall lowering

! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

!*** Test a FORALL construct with an array assignment
!    This is similar to the following embedded WHERE construct test, but the
!    elements are assigned unconditionally.
subroutine test_forall_with_array_assignment(aa,bb)
  type t
     integer(kind=8) :: block1(64)
     integer(kind=8) :: block2(64)
  end type t
  type(t) :: aa(10), bb(10)

  forall (i=1:10:2)
     aa(i)%block1 = bb(i+1)%block2
  end forall
end subroutine test_forall_with_array_assignment

! CHECK-LABEL: func.func @_QPtest_forall_with_array_assignment(
! CHECK-SAME:                                                  %[[VAL_0:.*]]: !fir.ref<!fir.array<10x!fir.type<_QFtest_forall_with_array_assignmentTt{block1:!fir.array<64xi64>,block2:!fir.array<64xi64>}>>> {fir.bindc_name = "aa"},
! CHECK-SAME:                                                  %[[VAL_1:.*]]: !fir.ref<!fir.array<10x!fir.type<_QFtest_forall_with_array_assignmentTt{block1:!fir.array<64xi64>,block2:!fir.array<64xi64>}>>> {fir.bindc_name = "bb"}) {
! CHECK:         %[[VAL_2:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:         %[[VAL_13:.*]] = arith.constant 10 : index
! CHECK:         %[[VAL_14:.*]] = fir.shape %[[VAL_13]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_15:.*]]:2 = hlfir.declare %[[VAL_0]](%[[VAL_14]]) dummy_scope %[[VAL_2]] arg 1 {uniq_name = "_QFtest_forall_with_array_assignmentEaa"} : (!fir.ref<!fir.array<10x!fir.type<_QFtest_forall_with_array_assignmentTt{block1:!fir.array<64xi64>,block2:!fir.array<64xi64>}>>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<10x!fir.type<_QFtest_forall_with_array_assignmentTt{block1:!fir.array<64xi64>,block2:!fir.array<64xi64>}>>>, !fir.ref<!fir.array<10x!fir.type<_QFtest_forall_with_array_assignmentTt{block1:!fir.array<64xi64>,block2:!fir.array<64xi64>}>>>)
! CHECK:         %[[VAL_16:.*]] = arith.constant 10 : index
! CHECK:         %[[VAL_17:.*]] = fir.shape %[[VAL_16]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_18:.*]]:2 = hlfir.declare %[[VAL_1]](%[[VAL_17]]) dummy_scope %[[VAL_2]] arg 2 {uniq_name = "_QFtest_forall_with_array_assignmentEbb"} : (!fir.ref<!fir.array<10x!fir.type<_QFtest_forall_with_array_assignmentTt{block1:!fir.array<64xi64>,block2:!fir.array<64xi64>}>>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<10x!fir.type<_QFtest_forall_with_array_assignmentTt{block1:!fir.array<64xi64>,block2:!fir.array<64xi64>}>>>, !fir.ref<!fir.array<10x!fir.type<_QFtest_forall_with_array_assignmentTt{block1:!fir.array<64xi64>,block2:!fir.array<64xi64>}>>>)
! CHECK:         %[[VAL_19:.*]] = arith.constant 1 : i32
! CHECK:         %[[VAL_20:.*]] = arith.constant 10 : i32
! CHECK:         %[[VAL_21:.*]] = arith.constant 2 : i32
! CHECK:         hlfir.forall lb {
! CHECK:           hlfir.yield %[[VAL_19]] : i32
! CHECK:         } ub {
! CHECK:           hlfir.yield %[[VAL_20]] : i32
! CHECK:         } step {
! CHECK:           hlfir.yield %[[VAL_21]] : i32
! CHECK:         }  (%[[VAL_22:.*]]: i32) {
! CHECK:           %[[VAL_23:.*]] = hlfir.forall_index "i" %[[VAL_22]] : (i32) -> !fir.ref<i32>
! CHECK:           hlfir.region_assign {
! CHECK:             %[[VAL_24:.*]] = fir.load %[[VAL_23]] : !fir.ref<i32>
! CHECK:             %[[VAL_25:.*]] = arith.constant 1 : i32
! CHECK:             %[[VAL_26:.*]] = arith.addi %[[VAL_24]], %[[VAL_25]] overflow<nsw> : i32
! CHECK:             %[[VAL_27:.*]] = fir.convert %[[VAL_26]] : (i32) -> i64
! CHECK:             %[[VAL_28:.*]] = hlfir.designate %[[VAL_18]]#0 (%[[VAL_27]])  : (!fir.ref<!fir.array<10x!fir.type<_QFtest_forall_with_array_assignmentTt{block1:!fir.array<64xi64>,block2:!fir.array<64xi64>}>>>, i64) -> !fir.ref<!fir.type<_QFtest_forall_with_array_assignmentTt{block1:!fir.array<64xi64>,block2:!fir.array<64xi64>}>>
! CHECK:             %[[VAL_29:.*]] = arith.constant 64 : index
! CHECK:             %[[VAL_30:.*]] = fir.shape %[[VAL_29]] : (index) -> !fir.shape<1>
! CHECK:             %[[VAL_31:.*]] = hlfir.designate %[[VAL_28]]{"block2"}{{ *}}shape %[[VAL_30]] : (!fir.ref<!fir.type<_QFtest_forall_with_array_assignmentTt{block1:!fir.array<64xi64>,block2:!fir.array<64xi64>}>>, !fir.shape<1>) -> !fir.ref<!fir.array<64xi64>>
! CHECK:             hlfir.yield %[[VAL_31]] : !fir.ref<!fir.array<64xi64>>
! CHECK:           } to {
! CHECK:             %[[VAL_32:.*]] = fir.load %[[VAL_23]] : !fir.ref<i32>
! CHECK:             %[[VAL_33:.*]] = fir.convert %[[VAL_32]] : (i32) -> i64
! CHECK:             %[[VAL_34:.*]] = hlfir.designate %[[VAL_15]]#0 (%[[VAL_33]])  : (!fir.ref<!fir.array<10x!fir.type<_QFtest_forall_with_array_assignmentTt{block1:!fir.array<64xi64>,block2:!fir.array<64xi64>}>>>, i64) -> !fir.ref<!fir.type<_QFtest_forall_with_array_assignmentTt{block1:!fir.array<64xi64>,block2:!fir.array<64xi64>}>>
! CHECK:             %[[VAL_35:.*]] = arith.constant 64 : index
! CHECK:             %[[VAL_36:.*]] = fir.shape %[[VAL_35]] : (index) -> !fir.shape<1>
! CHECK:             %[[VAL_37:.*]] = hlfir.designate %[[VAL_34]]{"block1"}{{ *}}shape %[[VAL_36]] : (!fir.ref<!fir.type<_QFtest_forall_with_array_assignmentTt{block1:!fir.array<64xi64>,block2:!fir.array<64xi64>}>>, !fir.shape<1>) -> !fir.ref<!fir.array<64xi64>>
! CHECK:             hlfir.yield %[[VAL_37]] : !fir.ref<!fir.array<64xi64>>
! CHECK:           }
! CHECK:         }
! CHECK:         return
! CHECK:       }

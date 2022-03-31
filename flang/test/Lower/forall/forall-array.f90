! Test forall lowering

! RUN: bbc -emit-fir %s -o - | FileCheck %s

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

! CHECK-LABEL: func @_QPtest_forall_with_array_assignment(
! CHECK-SAME:     %[[VAL_0:.*]]: !fir.ref<!fir.array<10x!fir.type<_QFtest_forall_with_array_assignmentTt{block1:!fir.array<64xi64>,block2:!fir.array<64xi64>}>>>{{.*}}, %[[VAL_1:.*]]: !fir.ref<!fir.array<10x!fir.type<_QFtest_forall_with_array_assignmentTt{block1:!fir.array<64xi64>,block2:!fir.array<64xi64>}>>>{{.*}}) {
! CHECK:         %[[VAL_2:.*]] = fir.alloca i32 {adapt.valuebyref, bindc_name = "i"}
! CHECK:         %[[VAL_3:.*]] = arith.constant 10 : index
! CHECK:         %[[VAL_4:.*]] = arith.constant 10 : index
! CHECK:         %[[VAL_5:.*]] = arith.constant 1 : i32
! CHECK:         %[[VAL_6:.*]] = fir.convert %[[VAL_5]] : (i32) -> index
! CHECK:         %[[VAL_7:.*]] = arith.constant 10 : i32
! CHECK:         %[[VAL_8:.*]] = fir.convert %[[VAL_7]] : (i32) -> index
! CHECK:         %[[VAL_9:.*]] = arith.constant 2 : i32
! CHECK:         %[[VAL_10:.*]] = fir.convert %[[VAL_9]] : (i32) -> index
! CHECK:         %[[VAL_11:.*]] = fir.shape %[[VAL_3]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_12:.*]] = fir.array_load %[[VAL_0]](%[[VAL_11]]) : (!fir.ref<!fir.array<10x!fir.type<_QFtest_forall_with_array_assignmentTt{block1:!fir.array<64xi64>,block2:!fir.array<64xi64>}>>>, !fir.shape<1>) -> !fir.array<10x!fir.type<_QFtest_forall_with_array_assignmentTt{block1:!fir.array<64xi64>,block2:!fir.array<64xi64>}>>
! CHECK:         %[[VAL_13:.*]] = fir.shape %[[VAL_4]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_14:.*]] = fir.array_load %[[VAL_1]](%[[VAL_13]]) : (!fir.ref<!fir.array<10x!fir.type<_QFtest_forall_with_array_assignmentTt{block1:!fir.array<64xi64>,block2:!fir.array<64xi64>}>>>, !fir.shape<1>) -> !fir.array<10x!fir.type<_QFtest_forall_with_array_assignmentTt{block1:!fir.array<64xi64>,block2:!fir.array<64xi64>}>>
! CHECK:         %[[VAL_15:.*]] = fir.do_loop %[[VAL_16:.*]] = %[[VAL_6]] to %[[VAL_8]] step %[[VAL_10]] unordered iter_args(%[[VAL_17:.*]] = %[[VAL_12]]) -> (!fir.array<10x!fir.type<_QFtest_forall_with_array_assignmentTt{block1:!fir.array<64xi64>,block2:!fir.array<64xi64>}>>) {
! CHECK:           %[[VAL_18:.*]] = fir.convert %[[VAL_16]] : (index) -> i32
! CHECK:           fir.store %[[VAL_18]] to %[[VAL_2]] : !fir.ref<i32>
! CHECK:           %[[VAL_19:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_20:.*]] = fir.load %[[VAL_2]] : !fir.ref<i32>
! CHECK:           %[[VAL_21:.*]] = fir.convert %[[VAL_20]] : (i32) -> i64
! CHECK:           %[[VAL_22:.*]] = fir.convert %[[VAL_21]] : (i64) -> index
! CHECK:           %[[VAL_23:.*]] = arith.subi %[[VAL_22]], %[[VAL_19]] : index
! CHECK:           %[[VAL_24:.*]] = fir.field_index block1, !fir.type<_QFtest_forall_with_array_assignmentTt{block1:!fir.array<64xi64>,block2:!fir.array<64xi64>}>
! CHECK:           %[[VAL_26:.*]] = arith.constant 64 : index
! CHECK:           %[[VAL_27:.*]] = arith.constant 1 : index
! CHECK-DAG:       %[[VAL_28:.*]] = fir.load %[[VAL_2]] : !fir.ref<i32>
! CHECK-DAG:       %[[VAL_29:.*]] = arith.constant 1 : i32
! CHECK:           %[[VAL_30:.*]] = arith.addi %[[VAL_28]], %[[VAL_29]] : i32
! CHECK:           %[[VAL_31:.*]] = fir.convert %[[VAL_30]] : (i32) -> i64
! CHECK:           %[[VAL_32:.*]] = fir.convert %[[VAL_31]] : (i64) -> index
! CHECK:           %[[VAL_33:.*]] = arith.subi %[[VAL_32]], %[[VAL_27]] : index
! CHECK:           %[[VAL_34:.*]] = fir.field_index block2, !fir.type<_QFtest_forall_with_array_assignmentTt{block1:!fir.array<64xi64>,block2:!fir.array<64xi64>}>
! CHECK:           %[[VAL_35:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_36:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_37:.*]] = arith.subi %[[VAL_26]], %[[VAL_35]] : index
! CHECK:           %[[VAL_38:.*]] = fir.do_loop %[[VAL_39:.*]] = %[[VAL_36]] to %[[VAL_37]] step %[[VAL_35]] unordered iter_args(%[[VAL_40:.*]] = %[[VAL_17]]) -> (!fir.array<10x!fir.type<_QFtest_forall_with_array_assignmentTt{block1:!fir.array<64xi64>,block2:!fir.array<64xi64>}>>) {
! CHECK:             %[[VAL_41:.*]] = fir.array_fetch %[[VAL_14]], %[[VAL_33]], %[[VAL_34]], %[[VAL_39]] : (!fir.array<10x!fir.type<_QFtest_forall_with_array_assignmentTt{block1:!fir.array<64xi64>,block2:!fir.array<64xi64>}>>, index, !fir.field, index) -> i64
! CHECK:             %[[VAL_42:.*]] = fir.array_update %[[VAL_40]], %[[VAL_41]], %[[VAL_23]], %[[VAL_24]], %[[VAL_39]] : (!fir.array<10x!fir.type<_QFtest_forall_with_array_assignmentTt{block1:!fir.array<64xi64>,block2:!fir.array<64xi64>}>>, i64, index, !fir.field, index) -> !fir.array<10x!fir.type<_QFtest_forall_with_array_assignmentTt{block1:!fir.array<64xi64>,block2:!fir.array<64xi64>}>>
! CHECK:             fir.result %[[VAL_42]] : !fir.array<10x!fir.type<_QFtest_forall_with_array_assignmentTt{block1:!fir.array<64xi64>,block2:!fir.array<64xi64>}>>
! CHECK:           }
! CHECK:           fir.result %[[VAL_43:.*]] : !fir.array<10x!fir.type<_QFtest_forall_with_array_assignmentTt{block1:!fir.array<64xi64>,block2:!fir.array<64xi64>}>>
! CHECK:         }
! CHECK:         fir.array_merge_store %[[VAL_12]], %[[VAL_44:.*]] to %[[VAL_0]] : !fir.array<10x!fir.type<_QFtest_forall_with_array_assignmentTt{block1:!fir.array<64xi64>,block2:!fir.array<64xi64>}>>, !fir.array<10x!fir.type<_QFtest_forall_with_array_assignmentTt{block1:!fir.array<64xi64>,block2:!fir.array<64xi64>}>>, !fir.ref<!fir.array<10x!fir.type<_QFtest_forall_with_array_assignmentTt{block1:!fir.array<64xi64>,block2:!fir.array<64xi64>}>>>
! CHECK:         return
! CHECK:       }

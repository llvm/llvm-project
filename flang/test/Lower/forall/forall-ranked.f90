! Test forall lowering

! RUN: bbc -emit-fir %s -o - | FileCheck %s

! CHECK-LABEL: func @_QPtest_forall_with_ranked_dimension() {
! CHECK:         %[[VAL_0:.*]] = fir.alloca i32 {adapt.valuebyref, bindc_name = "i"}
! CHECK:         %[[VAL_1:.*]] = arith.constant 10 : index
! CHECK:         %[[VAL_2:.*]] = arith.constant 10 : index
! CHECK:         %[[VAL_3:.*]] = fir.alloca !fir.array<10x10x!fir.type<_QFtest_forall_with_ranked_dimensionTt{arr:!fir.array<11xi32>}>> {bindc_name = "a", uniq_name = "_QFtest_forall_with_ranked_dimensionEa"}
! CHECK:         %[[VAL_4:.*]] = arith.constant 1 : i32
! CHECK:         %[[VAL_5:.*]] = fir.convert %[[VAL_4]] : (i32) -> index
! CHECK:         %[[VAL_6:.*]] = arith.constant 5 : i32
! CHECK:         %[[VAL_7:.*]] = fir.convert %[[VAL_6]] : (i32) -> index
! CHECK:         %[[VAL_8:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_9:.*]] = fir.shape %[[VAL_1]], %[[VAL_2]] : (index, index) -> !fir.shape<2>
! CHECK:         %[[VAL_10:.*]] = fir.array_load %[[VAL_3]](%[[VAL_9]]) : (!fir.ref<!fir.array<10x10x!fir.type<_QFtest_forall_with_ranked_dimensionTt{arr:!fir.array<11xi32>}>>>, !fir.shape<2>) -> !fir.array<10x10x!fir.type<_QFtest_forall_with_ranked_dimensionTt{arr:!fir.array<11xi32>}>>
! CHECK:         %[[VAL_11:.*]] = fir.do_loop %[[VAL_12:.*]] = %[[VAL_5]] to %[[VAL_7]] step %[[VAL_8]] unordered iter_args(%[[VAL_13:.*]] = %[[VAL_10]]) -> (!fir.array<10x10x!fir.type<_QFtest_forall_with_ranked_dimensionTt{arr:!fir.array<11xi32>}>>) {
! CHECK:           %[[VAL_14:.*]] = fir.convert %[[VAL_12]] : (index) -> i32
! CHECK:           fir.store %[[VAL_14]] to %[[VAL_0]] : !fir.ref<i32>
! CHECK:           %[[VAL_16:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_17:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_18:.*]] = fir.load %[[VAL_0]] : !fir.ref<i32>
! CHECK:           %[[VAL_19:.*]] = fir.convert %[[VAL_18]] : (i32) -> i64
! CHECK:           %[[VAL_20:.*]] = fir.convert %[[VAL_19]] : (i64) -> index
! CHECK:           %[[VAL_21:.*]] = arith.subi %[[VAL_20]], %[[VAL_17]] : index
! CHECK:           %[[VAL_22:.*]] = arith.constant 1 : i64
! CHECK:           %[[VAL_23:.*]] = fir.convert %[[VAL_22]] : (i64) -> index
! CHECK:           %[[VAL_24:.*]] = arith.addi %[[VAL_17]], %[[VAL_2]] : index
! CHECK:           %[[VAL_25:.*]] = arith.subi %[[VAL_24]], %[[VAL_17]] : index
! CHECK:           %[[VAL_26:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_27:.*]] = arith.subi %[[VAL_25]], %[[VAL_17]] : index
! CHECK:           %[[VAL_28:.*]] = arith.addi %[[VAL_27]], %[[VAL_23]] : index
! CHECK:           %[[VAL_29:.*]] = arith.divsi %[[VAL_28]], %[[VAL_23]] : index
! CHECK:           %[[VAL_30:.*]] = arith.cmpi sgt, %[[VAL_29]], %[[VAL_26]] : index
! CHECK:           %[[VAL_31:.*]] = arith.select %[[VAL_30]], %[[VAL_29]], %[[VAL_26]] : index
! CHECK:           %[[VAL_32:.*]] = fir.field_index arr, !fir.type<_QFtest_forall_with_ranked_dimensionTt{arr:!fir.array<11xi32>}>
! CHECK-DAG:       %[[VAL_33:.*]] = fir.load %[[VAL_0]] : !fir.ref<i32>
! CHECK-DAG:       %[[VAL_34:.*]] = arith.constant 4 : i32
! CHECK:           %[[VAL_35:.*]] = arith.addi %[[VAL_33]], %[[VAL_34]] : i32
! CHECK:           %[[VAL_36:.*]] = fir.convert %[[VAL_35]] : (i32) -> i64
! CHECK:           %[[VAL_37:.*]] = fir.convert %[[VAL_36]] : (i64) -> index
! CHECK:           %[[VAL_38:.*]] = arith.subi %[[VAL_37]], %[[VAL_16]] : index
! CHECK:           %[[VAL_39:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_40:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_41:.*]] = arith.subi %[[VAL_31]], %[[VAL_39]] : index
! CHECK:           %[[VAL_42:.*]] = fir.do_loop %[[VAL_43:.*]] = %[[VAL_40]] to %[[VAL_41]] step %[[VAL_39]] unordered iter_args(%[[VAL_44:.*]] = %[[VAL_13]]) -> (!fir.array<10x10x!fir.type<_QFtest_forall_with_ranked_dimensionTt{arr:!fir.array<11xi32>}>>) {
! CHECK:             %[[VAL_45:.*]] = fir.call @_QPf(%[[VAL_0]]) : (!fir.ref<i32>) -> i32
! CHECK:             %[[VAL_46:.*]] = arith.subi %[[VAL_17]], %[[VAL_17]] : index
! CHECK:             %[[VAL_47:.*]] = arith.muli %[[VAL_43]], %[[VAL_23]] : index
! CHECK:             %[[VAL_48:.*]] = arith.addi %[[VAL_46]], %[[VAL_47]] : index
! CHECK:             %[[VAL_49:.*]] = fir.array_update %[[VAL_44]], %[[VAL_45]], %[[VAL_21]], %[[VAL_48]], %[[VAL_32]], %[[VAL_38]] : (!fir.array<10x10x!fir.type<_QFtest_forall_with_ranked_dimensionTt{arr:!fir.array<11xi32>}>>, i32, index, index, !fir.field, index) -> !fir.array<10x10x!fir.type<_QFtest_forall_with_ranked_dimensionTt{arr:!fir.array<11xi32>}>>
! CHECK:             fir.result %[[VAL_49]] : !fir.array<10x10x!fir.type<_QFtest_forall_with_ranked_dimensionTt{arr:!fir.array<11xi32>}>>
! CHECK:           }
! CHECK:           fir.result %[[VAL_50:.*]] : !fir.array<10x10x!fir.type<_QFtest_forall_with_ranked_dimensionTt{arr:!fir.array<11xi32>}>>
! CHECK:         }
! CHECK:         fir.array_merge_store %[[VAL_10]], %[[VAL_51:.*]] to %[[VAL_3]] : !fir.array<10x10x!fir.type<_QFtest_forall_with_ranked_dimensionTt{arr:!fir.array<11xi32>}>>, !fir.array<10x10x!fir.type<_QFtest_forall_with_ranked_dimensionTt{arr:!fir.array<11xi32>}>>, !fir.ref<!fir.array<10x10x!fir.type<_QFtest_forall_with_ranked_dimensionTt{arr:!fir.array<11xi32>}>>>
! CHECK:         return
! CHECK:       }

subroutine test_forall_with_ranked_dimension
  interface
     pure integer function f(i)
       integer, intent(in) :: i
     end function f
  end interface
  type t
     !integer :: arr(5:15)
     integer :: arr(11)
  end type t
  type(t) :: a(10,10)
  
  forall (i=1:5)
     a(i,:)%arr(i+4) = f(i)
  end forall
end subroutine test_forall_with_ranked_dimension

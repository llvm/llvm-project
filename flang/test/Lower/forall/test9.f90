! Test forall lowering

! RUN: bbc -emit-fir %s -o - | FileCheck %s

!*** This FORALL construct does present a potential loop-carried dependence if
!*** implemented naively (and incorrectly). The final value of a(3) must be the
!*** value of a(2) before loopy begins execution added to b(2).
subroutine test9(a,b,n)

  integer :: n
  real, intent(inout) :: a(n)
  real, intent(in) :: b(n)
  loopy: FORALL (i=1:n-1)
     a(i+1) = a(i) + b(i)
  END FORALL loopy
end subroutine test9

! CHECK-LABEL: func @_QPtest9(
! CHECK-SAME:    %[[VAL_0:.*]]: !fir.ref<!fir.array<?xf32>>{{.*}}, %[[VAL_1:.*]]: !fir.ref<!fir.array<?xf32>>{{.*}}, %[[VAL_2:.*]]: !fir.ref<i32>{{.*}}) {
! CHECK:         %[[VAL_3:.*]] = fir.alloca i32 {adapt.valuebyref, bindc_name = "i"}
! CHECK:         %[[VAL_4:.*]] = fir.load %[[VAL_2]] : !fir.ref<i32>
! CHECK:         %[[VAL_5:.*]] = fir.convert %[[VAL_4]] : (i32) -> i64
! CHECK:         %[[VAL_6A:.*]] = fir.convert %[[VAL_5]] : (i64) -> index
! CHECK:         %[[C0:.*]] = arith.constant 0 : index
! CHECK:         %[[CMP:.*]] = arith.cmpi sgt, %[[VAL_6A]], %[[C0]] : index
! CHECK:         %[[VAL_6:.*]] = arith.select %[[CMP]], %[[VAL_6A]], %[[C0]] : index
! CHECK:         %[[VAL_7:.*]] = fir.load %[[VAL_2]] : !fir.ref<i32>
! CHECK:         %[[VAL_8:.*]] = fir.convert %[[VAL_7]] : (i32) -> i64
! CHECK:         %[[VAL_9A:.*]] = fir.convert %[[VAL_8]] : (i64) -> index
! CHECK:         %[[C0_2:.*]] = arith.constant 0 : index
! CHECK:         %[[CMP_2:.*]] = arith.cmpi sgt, %[[VAL_9A]], %[[C0_2]] : index
! CHECK:         %[[VAL_9:.*]] = arith.select %[[CMP_2]], %[[VAL_9A]], %[[C0_2]] : index
! CHECK:         %[[VAL_10:.*]] = arith.constant 1 : i32
! CHECK:         %[[VAL_11:.*]] = fir.convert %[[VAL_10]] : (i32) -> index
! CHECK:         %[[VAL_12:.*]] = fir.load %[[VAL_2]] : !fir.ref<i32>
! CHECK:         %[[VAL_13:.*]] = arith.constant 1 : i32
! CHECK:         %[[VAL_14:.*]] = arith.subi %[[VAL_12]], %[[VAL_13]] : i32
! CHECK:         %[[VAL_15:.*]] = fir.convert %[[VAL_14]] : (i32) -> index
! CHECK:         %[[VAL_16:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_17:.*]] = fir.shape %[[VAL_6]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_18:.*]] = fir.array_load %[[VAL_0]](%[[VAL_17]]) : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.array<?xf32>
! CHECK:         %[[VAL_19:.*]] = fir.shape %[[VAL_6]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_20:.*]] = fir.array_load %[[VAL_0]](%[[VAL_19]]) : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.array<?xf32>
! CHECK:         %[[VAL_21:.*]] = fir.shape %[[VAL_9]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_22:.*]] = fir.array_load %[[VAL_1]](%[[VAL_21]]) : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.array<?xf32>
! CHECK:         %[[VAL_23:.*]] = fir.do_loop %[[VAL_24:.*]] = %[[VAL_11]] to %[[VAL_15]] step %[[VAL_16]] unordered iter_args(%[[VAL_25:.*]] = %[[VAL_18]]) -> (!fir.array<?xf32>) {
! CHECK:           %[[VAL_26:.*]] = fir.convert %[[VAL_24]] : (index) -> i32
! CHECK:           fir.store %[[VAL_26]] to %[[VAL_3]] : !fir.ref<i32>
! CHECK:           %[[VAL_27:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_28:.*]] = fir.load %[[VAL_3]] : !fir.ref<i32>
! CHECK:           %[[VAL_29:.*]] = fir.convert %[[VAL_28]] : (i32) -> i64
! CHECK:           %[[VAL_30:.*]] = fir.convert %[[VAL_29]] : (i64) -> index
! CHECK:           %[[VAL_31:.*]] = arith.subi %[[VAL_30]], %[[VAL_27]] : index
! CHECK:           %[[VAL_32:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_33:.*]] = fir.load %[[VAL_3]] : !fir.ref<i32>
! CHECK:           %[[VAL_34:.*]] = fir.convert %[[VAL_33]] : (i32) -> i64
! CHECK:           %[[VAL_35:.*]] = fir.convert %[[VAL_34]] : (i64) -> index
! CHECK:           %[[VAL_36:.*]] = arith.subi %[[VAL_35]], %[[VAL_32]] : index
! CHECK:           %[[VAL_37:.*]] = fir.array_fetch %[[VAL_20]], %[[VAL_31]] : (!fir.array<?xf32>, index) -> f32
! CHECK:           %[[VAL_38:.*]] = fir.array_fetch %[[VAL_22]], %[[VAL_36]] : (!fir.array<?xf32>, index) -> f32
! CHECK:           %[[VAL_39:.*]] = arith.addf %[[VAL_37]], %[[VAL_38]] : f32
! CHECK:           %[[VAL_40:.*]] = arith.constant 1 : index
! CHECK-DAG:       %[[VAL_41:.*]] = fir.load %[[VAL_3]] : !fir.ref<i32>
! CHECK-DAG:       %[[VAL_42:.*]] = arith.constant 1 : i32
! CHECK:           %[[VAL_43:.*]] = arith.addi %[[VAL_41]], %[[VAL_42]] : i32
! CHECK:           %[[VAL_44:.*]] = fir.convert %[[VAL_43]] : (i32) -> i64
! CHECK:           %[[VAL_45:.*]] = fir.convert %[[VAL_44]] : (i64) -> index
! CHECK:           %[[VAL_46:.*]] = arith.subi %[[VAL_45]], %[[VAL_40]] : index
! CHECK:           %[[VAL_47:.*]] = fir.array_update %[[VAL_25]], %[[VAL_39]], %[[VAL_46]] : (!fir.array<?xf32>, f32, index) -> !fir.array<?xf32>
! CHECK:           fir.result %[[VAL_47]] : !fir.array<?xf32>
! CHECK:         }
! CHECK:         fir.array_merge_store %[[VAL_18]], %[[VAL_48:.*]] to %[[VAL_0]] : !fir.array<?xf32>, !fir.array<?xf32>, !fir.ref<!fir.array<?xf32>>
! CHECK:         return
! CHECK:       }

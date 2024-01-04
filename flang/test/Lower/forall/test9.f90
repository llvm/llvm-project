! Test forall lowering

! RUN: bbc -emit-fir -hlfir=false %s -o - | FileCheck %s

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
! CHECK-SAME:                 %[[VAL_0:.*]]: !fir.ref<!fir.array<?xf32>> {fir.bindc_name = "a"},
! CHECK-SAME:                 %[[VAL_1:.*]]: !fir.ref<!fir.array<?xf32>> {fir.bindc_name = "b"},
! CHECK-SAME:                 %[[VAL_2:.*]]: !fir.ref<i32> {fir.bindc_name = "n"}) {
! CHECK:         %[[VAL_3:.*]] = fir.alloca i32 {adapt.valuebyref, bindc_name = "i"}
! CHECK:         %[[VAL_4:.*]] = fir.load %[[VAL_2]] : !fir.ref<i32>
! CHECK:         %[[VAL_5:.*]] = fir.convert %[[VAL_4]] : (i32) -> i64
! CHECK:         %[[VAL_6:.*]] = fir.convert %[[VAL_5]] : (i64) -> index
! CHECK:         %[[VAL_7:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_8:.*]] = arith.cmpi sgt, %[[VAL_6]], %[[VAL_7]] : index
! CHECK:         %[[VAL_9:.*]] = arith.select %[[VAL_8]], %[[VAL_6]], %[[VAL_7]] : index
! CHECK:         %[[VAL_10:.*]] = fir.load %[[VAL_2]] : !fir.ref<i32>
! CHECK:         %[[VAL_11:.*]] = fir.convert %[[VAL_10]] : (i32) -> i64
! CHECK:         %[[VAL_12:.*]] = fir.convert %[[VAL_11]] : (i64) -> index
! CHECK:         %[[VAL_13:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_14:.*]] = arith.cmpi sgt, %[[VAL_12]], %[[VAL_13]] : index
! CHECK:         %[[VAL_15:.*]] = arith.select %[[VAL_14]], %[[VAL_12]], %[[VAL_13]] : index
! CHECK:         %[[VAL_16:.*]] = arith.constant 1 : i32
! CHECK:         %[[VAL_17:.*]] = fir.convert %[[VAL_16]] : (i32) -> index
! CHECK:         %[[VAL_18:.*]] = fir.load %[[VAL_2]] : !fir.ref<i32>
! CHECK:         %[[VAL_19:.*]] = arith.constant 1 : i32
! CHECK:         %[[VAL_20:.*]] = arith.subi %[[VAL_18]], %[[VAL_19]] : i32
! CHECK:         %[[VAL_21:.*]] = fir.convert %[[VAL_20]] : (i32) -> index
! CHECK:         %[[VAL_22:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_23:.*]] = fir.shape %[[VAL_9]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_24:.*]] = fir.array_load %[[VAL_0]](%[[VAL_23]]) : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.array<?xf32>
! CHECK:         %[[VAL_25:.*]] = fir.shape %[[VAL_9]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_26:.*]] = fir.array_load %[[VAL_0]](%[[VAL_25]]) : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.array<?xf32>
! CHECK:         %[[VAL_27:.*]] = fir.shape %[[VAL_15]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_28:.*]] = fir.array_load %[[VAL_1]](%[[VAL_27]]) : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.array<?xf32>
! CHECK:         %[[VAL_29:.*]] = fir.do_loop %[[VAL_30:.*]] = %[[VAL_17]] to %[[VAL_21]] step %[[VAL_22]] unordered iter_args(%[[VAL_31:.*]] = %[[VAL_24]]) -> (!fir.array<?xf32>) {
! CHECK:           %[[VAL_32:.*]] = fir.convert %[[VAL_30]] : (index) -> i32
! CHECK:           fir.store %[[VAL_32]] to %[[VAL_3]] : !fir.ref<i32>
! CHECK:           %[[VAL_33:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_34:.*]] = fir.load %[[VAL_3]] : !fir.ref<i32>
! CHECK:           %[[VAL_35:.*]] = fir.convert %[[VAL_34]] : (i32) -> i64
! CHECK:           %[[VAL_36:.*]] = fir.convert %[[VAL_35]] : (i64) -> index
! CHECK:           %[[VAL_37:.*]] = arith.subi %[[VAL_36]], %[[VAL_33]] : index
! CHECK:           %[[VAL_38:.*]] = fir.array_fetch %[[VAL_26]], %[[VAL_37]] : (!fir.array<?xf32>, index) -> f32
! CHECK:           %[[VAL_39:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_40:.*]] = fir.load %[[VAL_3]] : !fir.ref<i32>
! CHECK:           %[[VAL_41:.*]] = fir.convert %[[VAL_40]] : (i32) -> i64
! CHECK:           %[[VAL_42:.*]] = fir.convert %[[VAL_41]] : (i64) -> index
! CHECK:           %[[VAL_43:.*]] = arith.subi %[[VAL_42]], %[[VAL_39]] : index
! CHECK:           %[[VAL_44:.*]] = fir.array_fetch %[[VAL_28]], %[[VAL_43]] : (!fir.array<?xf32>, index) -> f32
! CHECK:           %[[VAL_45:.*]] = arith.addf %[[VAL_38]], %[[VAL_44]] {{.*}}: f32
! CHECK:           %[[VAL_46:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_47:.*]] = fir.load %[[VAL_3]] : !fir.ref<i32>
! CHECK:           %[[VAL_48:.*]] = arith.constant 1 : i32
! CHECK:           %[[VAL_49:.*]] = arith.addi %[[VAL_47]], %[[VAL_48]] : i32
! CHECK:           %[[VAL_50:.*]] = fir.convert %[[VAL_49]] : (i32) -> i64
! CHECK:           %[[VAL_51:.*]] = fir.convert %[[VAL_50]] : (i64) -> index
! CHECK:           %[[VAL_52:.*]] = arith.subi %[[VAL_51]], %[[VAL_46]] : index
! CHECK:           %[[VAL_53:.*]] = fir.array_update %[[VAL_31]], %[[VAL_45]], %[[VAL_52]] : (!fir.array<?xf32>, f32, index) -> !fir.array<?xf32>
! CHECK:           fir.result %[[VAL_53]] : !fir.array<?xf32>
! CHECK:         }
! CHECK:         fir.array_merge_store %[[VAL_24]], %[[VAL_54:.*]] to %[[VAL_0]] : !fir.array<?xf32>, !fir.array<?xf32>, !fir.ref<!fir.array<?xf32>>
! CHECK:         return
! CHECK:       }

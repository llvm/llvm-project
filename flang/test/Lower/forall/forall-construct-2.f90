! Test forall lowering

! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

!*** Test forall with multiple assignment statements
subroutine test2_forall_construct(a,b)
  real :: a(100,400), b(200,200)
  forall (i=1:100, j=1:200)
     a(i,j) = b(i,j) + b(i+1,j)
     a(i,200+j) = 1.0 / b(j, i)
  end forall
end subroutine test2_forall_construct

! CHECK-LABEL: func.func @_QPtest2_forall_construct(
! CHECK-SAME:                                       %[[VAL_0:.*]]: !fir.ref<!fir.array<100x400xf32>> {fir.bindc_name = "a"},
! CHECK-SAME:                                       %[[VAL_1:.*]]: !fir.ref<!fir.array<200x200xf32>> {fir.bindc_name = "b"}) {
! CHECK:         %[[VAL_2:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:         %[[VAL_3:.*]] = arith.constant 100 : index
! CHECK:         %[[VAL_4:.*]] = arith.constant 400 : index
! CHECK:         %[[VAL_5:.*]] = fir.shape %[[VAL_3]], %[[VAL_4]] : (index, index) -> !fir.shape<2>
! CHECK:         %[[VAL_6:.*]]:2 = hlfir.declare %[[VAL_0]](%[[VAL_5]]) dummy_scope %[[VAL_2]] arg 1 {uniq_name = "_QFtest2_forall_constructEa"} : (!fir.ref<!fir.array<100x400xf32>>, !fir.shape<2>, !fir.dscope) -> (!fir.ref<!fir.array<100x400xf32>>, !fir.ref<!fir.array<100x400xf32>>)
! CHECK:         %[[VAL_7:.*]] = arith.constant 200 : index
! CHECK:         %[[VAL_8:.*]] = arith.constant 200 : index
! CHECK:         %[[VAL_9:.*]] = fir.shape %[[VAL_7]], %[[VAL_8]] : (index, index) -> !fir.shape<2>
! CHECK:         %[[VAL_10:.*]]:2 = hlfir.declare %[[VAL_1]](%[[VAL_9]]) dummy_scope %[[VAL_2]] arg 2 {uniq_name = "_QFtest2_forall_constructEb"} : (!fir.ref<!fir.array<200x200xf32>>, !fir.shape<2>, !fir.dscope) -> (!fir.ref<!fir.array<200x200xf32>>, !fir.ref<!fir.array<200x200xf32>>)
! CHECK:         %[[VAL_11:.*]] = arith.constant 1 : i32
! CHECK:         %[[VAL_12:.*]] = arith.constant 100 : i32
! CHECK:         %[[VAL_15:.*]] = arith.constant 1 : i32
! CHECK:         %[[VAL_16:.*]] = arith.constant 200 : i32
! CHECK:         hlfir.forall lb {
! CHECK-NEXT:      hlfir.yield %[[VAL_11]] : i32
! CHECK-NEXT:    } ub {
! CHECK-NEXT:      hlfir.yield %[[VAL_12]] : i32
! CHECK-NEXT:    } (%[[VAL_13:.*]]: i32) {
! CHECK-NEXT:      %[[VAL_14:.*]] = hlfir.forall_index "i" %[[VAL_13]] : (i32) -> !fir.ref<i32>
! CHECK-NEXT:      hlfir.forall lb {
! CHECK-NEXT:        hlfir.yield %[[VAL_15]] : i32
! CHECK-NEXT:      } ub {
! CHECK-NEXT:        hlfir.yield %[[VAL_16]] : i32
! CHECK-NEXT:      } (%[[VAL_17:.*]]: i32) {
! CHECK-NEXT:        %[[VAL_18:.*]] = hlfir.forall_index "j" %[[VAL_17]] : (i32) -> !fir.ref<i32>
! CHECK-NEXT:        hlfir.region_assign {
! CHECK-NEXT:          %[[VAL_19:.*]] = fir.load %[[VAL_14]] : !fir.ref<i32>
! CHECK-NEXT:          %[[VAL_20:.*]] = fir.convert %[[VAL_19]] : (i32) -> i64
! CHECK-NEXT:          %[[VAL_21:.*]] = fir.load %[[VAL_18]] : !fir.ref<i32>
! CHECK-NEXT:          %[[VAL_22:.*]] = fir.convert %[[VAL_21]] : (i32) -> i64
! CHECK-NEXT:          %[[VAL_23:.*]] = hlfir.designate %[[VAL_10]]#0 (%[[VAL_20]], %[[VAL_22]])  : (!fir.ref<!fir.array<200x200xf32>>, i64, i64) -> !fir.ref<f32>
! CHECK-NEXT:          %[[VAL_24:.*]] = fir.load %[[VAL_23]] : !fir.ref<f32>
! CHECK-NEXT:          %[[VAL_25:.*]] = fir.load %[[VAL_14]] : !fir.ref<i32>
! CHECK-NEXT:          %[[VAL_26:.*]] = arith.constant 1 : i32
! CHECK-NEXT:          %[[VAL_27:.*]] = arith.addi %[[VAL_25]], %[[VAL_26]] overflow<nsw> : i32
! CHECK-NEXT:          %[[VAL_28:.*]] = fir.convert %[[VAL_27]] : (i32) -> i64
! CHECK-NEXT:          %[[VAL_29:.*]] = fir.load %[[VAL_18]] : !fir.ref<i32>
! CHECK-NEXT:          %[[VAL_30:.*]] = fir.convert %[[VAL_29]] : (i32) -> i64
! CHECK-NEXT:          %[[VAL_31:.*]] = hlfir.designate %[[VAL_10]]#0 (%[[VAL_28]], %[[VAL_30]])  : (!fir.ref<!fir.array<200x200xf32>>, i64, i64) -> !fir.ref<f32>
! CHECK-NEXT:          %[[VAL_32:.*]] = fir.load %[[VAL_31]] : !fir.ref<f32>
! CHECK-NEXT:          %[[VAL_33:.*]] = arith.addf %[[VAL_24]], %[[VAL_32]] fastmath<contract> : f32
! CHECK-NEXT:          hlfir.yield %[[VAL_33]] : f32
! CHECK-NEXT:        } to {
! CHECK-NEXT:          %[[VAL_34:.*]] = fir.load %[[VAL_14]] : !fir.ref<i32>
! CHECK-NEXT:          %[[VAL_35:.*]] = fir.convert %[[VAL_34]] : (i32) -> i64
! CHECK-NEXT:          %[[VAL_36:.*]] = fir.load %[[VAL_18]] : !fir.ref<i32>
! CHECK-NEXT:          %[[VAL_37:.*]] = fir.convert %[[VAL_36]] : (i32) -> i64
! CHECK-NEXT:          %[[VAL_38:.*]] = hlfir.designate %[[VAL_6]]#0 (%[[VAL_35]], %[[VAL_37]])  : (!fir.ref<!fir.array<100x400xf32>>, i64, i64) -> !fir.ref<f32>
! CHECK-NEXT:          hlfir.yield %[[VAL_38]] : !fir.ref<f32>
! CHECK-NEXT:        }
! CHECK-NEXT:        hlfir.region_assign {
! CHECK-NEXT:          %[[VAL_39:.*]] = arith.constant 1.000000e+00 : f32
! CHECK-NEXT:          %[[VAL_40:.*]] = fir.load %[[VAL_18]] : !fir.ref<i32>
! CHECK-NEXT:          %[[VAL_41:.*]] = fir.convert %[[VAL_40]] : (i32) -> i64
! CHECK-NEXT:          %[[VAL_42:.*]] = fir.load %[[VAL_14]] : !fir.ref<i32>
! CHECK-NEXT:          %[[VAL_43:.*]] = fir.convert %[[VAL_42]] : (i32) -> i64
! CHECK-NEXT:          %[[VAL_44:.*]] = hlfir.designate %[[VAL_10]]#0 (%[[VAL_41]], %[[VAL_43]])  : (!fir.ref<!fir.array<200x200xf32>>, i64, i64) -> !fir.ref<f32>
! CHECK-NEXT:          %[[VAL_45:.*]] = fir.load %[[VAL_44]] : !fir.ref<f32>
! CHECK-NEXT:          %[[VAL_46:.*]] = arith.divf %[[VAL_39]], %[[VAL_45]] fastmath<contract> : f32
! CHECK-NEXT:          hlfir.yield %[[VAL_46]] : f32
! CHECK-NEXT:        } to {
! CHECK-NEXT:          %[[VAL_47:.*]] = fir.load %[[VAL_14]] : !fir.ref<i32>
! CHECK-NEXT:          %[[VAL_48:.*]] = fir.convert %[[VAL_47]] : (i32) -> i64
! CHECK-NEXT:          %[[VAL_49:.*]] = arith.constant 200 : i32
! CHECK-NEXT:          %[[VAL_50:.*]] = fir.load %[[VAL_18]] : !fir.ref<i32>
! CHECK-NEXT:          %[[VAL_51:.*]] = arith.addi %[[VAL_49]], %[[VAL_50]] overflow<nsw> : i32
! CHECK-NEXT:          %[[VAL_52:.*]] = fir.convert %[[VAL_51]] : (i32) -> i64
! CHECK-NEXT:          %[[VAL_53:.*]] = hlfir.designate %[[VAL_6]]#0 (%[[VAL_48]], %[[VAL_52]])  : (!fir.ref<!fir.array<100x400xf32>>, i64, i64) -> !fir.ref<f32>
! CHECK-NEXT:          hlfir.yield %[[VAL_53]] : !fir.ref<f32>
! CHECK-NEXT:        }
! CHECK-NEXT:      }
! CHECK-NEXT:    }
! CHECK-NEXT:    return
! CHECK-NEXT:  }

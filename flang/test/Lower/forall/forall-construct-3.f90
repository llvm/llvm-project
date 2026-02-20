! Test forall lowering

! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

!*** Test forall with multiple assignment statements and mask
subroutine test3_forall_construct(a,b, mask)
  real :: a(100,400), b(200,200)
  logical :: mask(100,200)
  forall (i=1:100, j=1:200, mask(i,j))
     a(i,j) = b(i,j) + b(i+1,j)
     a(i,200+j) = 1.0 / b(j, i)
  end forall
end subroutine test3_forall_construct

! CHECK-LABEL: func.func @_QPtest3_forall_construct(
! CHECK-SAME:                                       %[[VAL_0:.*]]: !fir.ref<!fir.array<100x400xf32>> {fir.bindc_name = "a"},
! CHECK-SAME:                                       %[[VAL_1:.*]]: !fir.ref<!fir.array<200x200xf32>> {fir.bindc_name = "b"},
! CHECK-SAME:                                       %[[VAL_2:.*]]: !fir.ref<!fir.array<100x200x!fir.logical<4>>> {fir.bindc_name = "mask"}) {
! CHECK:         %[[VAL_3:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:         %[[VAL_4:.*]] = arith.constant 100 : index
! CHECK:         %[[VAL_5:.*]] = arith.constant 400 : index
! CHECK:         %[[VAL_6:.*]] = fir.shape %[[VAL_4]], %[[VAL_5]] : (index, index) -> !fir.shape<2>
! CHECK:         %[[VAL_7:.*]]:2 = hlfir.declare %[[VAL_0]](%[[VAL_6]]) dummy_scope %[[VAL_3]] arg 1 {uniq_name = "_QFtest3_forall_constructEa"} : (!fir.ref<!fir.array<100x400xf32>>, !fir.shape<2>, !fir.dscope) -> (!fir.ref<!fir.array<100x400xf32>>, !fir.ref<!fir.array<100x400xf32>>)
! CHECK:         %[[VAL_8:.*]] = arith.constant 200 : index
! CHECK:         %[[VAL_9:.*]] = arith.constant 200 : index
! CHECK:         %[[VAL_10:.*]] = fir.shape %[[VAL_8]], %[[VAL_9]] : (index, index) -> !fir.shape<2>
! CHECK:         %[[VAL_11:.*]]:2 = hlfir.declare %[[VAL_1]](%[[VAL_10]]) dummy_scope %[[VAL_3]] arg 2 {uniq_name = "_QFtest3_forall_constructEb"} : (!fir.ref<!fir.array<200x200xf32>>, !fir.shape<2>, !fir.dscope) -> (!fir.ref<!fir.array<200x200xf32>>, !fir.ref<!fir.array<200x200xf32>>)
! CHECK:         %[[VAL_12:.*]] = arith.constant 100 : index
! CHECK:         %[[VAL_13:.*]] = arith.constant 200 : index
! CHECK:         %[[VAL_14:.*]] = fir.shape %[[VAL_12]], %[[VAL_13]] : (index, index) -> !fir.shape<2>
! CHECK:         %[[VAL_15:.*]]:2 = hlfir.declare %[[VAL_2]](%[[VAL_14]]) dummy_scope %[[VAL_3]] arg 3 {uniq_name = "_QFtest3_forall_constructEmask"} : (!fir.ref<!fir.array<100x200x!fir.logical<4>>>, !fir.shape<2>, !fir.dscope) -> (!fir.ref<!fir.array<100x200x!fir.logical<4>>>, !fir.ref<!fir.array<100x200x!fir.logical<4>>>)
! CHECK:         %[[VAL_16:.*]] = arith.constant 1 : i32
! CHECK:         %[[VAL_17:.*]] = arith.constant 100 : i32
! CHECK:         %[[VAL_18:.*]] = arith.constant 1 : i32
! CHECK:         %[[VAL_19:.*]] = arith.constant 200 : i32
! CHECK:         hlfir.forall lb {
! CHECK:           hlfir.yield %[[VAL_16]] : i32
! CHECK:         } ub {
! CHECK:           hlfir.yield %[[VAL_17]] : i32
! CHECK:         } (%[[VAL_20:.*]]: i32) {
! CHECK:           %[[VAL_21:.*]] = hlfir.forall_index "i" %[[VAL_20]] : (i32) -> !fir.ref<i32>
! CHECK:           hlfir.forall lb {
! CHECK:             hlfir.yield %[[VAL_18]] : i32
! CHECK:           } ub {
! CHECK:             hlfir.yield %[[VAL_19]] : i32
! CHECK:           } (%[[VAL_22:.*]]: i32) {
! CHECK:             %[[VAL_23:.*]] = hlfir.forall_index "j" %[[VAL_22]] : (i32) -> !fir.ref<i32>
! CHECK:             hlfir.forall_mask {
! CHECK:               %[[VAL_24:.*]] = fir.load %[[VAL_21]] : !fir.ref<i32>
! CHECK:               %[[VAL_25:.*]] = fir.convert %[[VAL_24]] : (i32) -> i64
! CHECK:               %[[VAL_26:.*]] = fir.load %[[VAL_23]] : !fir.ref<i32>
! CHECK:               %[[VAL_27:.*]] = fir.convert %[[VAL_26]] : (i32) -> i64
! CHECK:               %[[VAL_28:.*]] = hlfir.designate %[[VAL_15]]#0 (%[[VAL_25]], %[[VAL_27]])  : (!fir.ref<!fir.array<100x200x!fir.logical<4>>>, i64, i64) -> !fir.ref<!fir.logical<4>>
! CHECK:               %[[VAL_29:.*]] = fir.load %[[VAL_28]] : !fir.ref<!fir.logical<4>>
! CHECK:               %[[VAL_30:.*]] = fir.convert %[[VAL_29]] : (!fir.logical<4>) -> i1
! CHECK:               hlfir.yield %[[VAL_30]] : i1
! CHECK:             } do {
! CHECK:               hlfir.region_assign {
! CHECK:                 %[[VAL_31:.*]] = fir.load %[[VAL_21]] : !fir.ref<i32>
! CHECK:                 %[[VAL_32:.*]] = fir.convert %[[VAL_31]] : (i32) -> i64
! CHECK:                 %[[VAL_33:.*]] = fir.load %[[VAL_23]] : !fir.ref<i32>
! CHECK:                 %[[VAL_34:.*]] = fir.convert %[[VAL_33]] : (i32) -> i64
! CHECK:                 %[[VAL_35:.*]] = hlfir.designate %[[VAL_11]]#0 (%[[VAL_32]], %[[VAL_34]])  : (!fir.ref<!fir.array<200x200xf32>>, i64, i64) -> !fir.ref<f32>
! CHECK:                 %[[VAL_36:.*]] = fir.load %[[VAL_35]] : !fir.ref<f32>
! CHECK:                 %[[VAL_37:.*]] = fir.load %[[VAL_21]] : !fir.ref<i32>
! CHECK:                 %[[VAL_38:.*]] = arith.constant 1 : i32
! CHECK:                 %[[VAL_39:.*]] = arith.addi %[[VAL_37]], %[[VAL_38]] overflow<nsw> : i32
! CHECK:                 %[[VAL_40:.*]] = fir.convert %[[VAL_39]] : (i32) -> i64
! CHECK:                 %[[VAL_41:.*]] = fir.load %[[VAL_23]] : !fir.ref<i32>
! CHECK:                 %[[VAL_42:.*]] = fir.convert %[[VAL_41]] : (i32) -> i64
! CHECK:                 %[[VAL_43:.*]] = hlfir.designate %[[VAL_11]]#0 (%[[VAL_40]], %[[VAL_42]])  : (!fir.ref<!fir.array<200x200xf32>>, i64, i64) -> !fir.ref<f32>
! CHECK:                 %[[VAL_44:.*]] = fir.load %[[VAL_43]] : !fir.ref<f32>
! CHECK:                 %[[VAL_45:.*]] = arith.addf %[[VAL_36]], %[[VAL_44]] fastmath<contract> : f32
! CHECK:                 hlfir.yield %[[VAL_45]] : f32
! CHECK:               } to {
! CHECK:                 %[[VAL_46:.*]] = fir.load %[[VAL_21]] : !fir.ref<i32>
! CHECK:                 %[[VAL_47:.*]] = fir.convert %[[VAL_46]] : (i32) -> i64
! CHECK:                 %[[VAL_48:.*]] = fir.load %[[VAL_23]] : !fir.ref<i32>
! CHECK:                 %[[VAL_49:.*]] = fir.convert %[[VAL_48]] : (i32) -> i64
! CHECK:                 %[[VAL_50:.*]] = hlfir.designate %[[VAL_7]]#0 (%[[VAL_47]], %[[VAL_49]])  : (!fir.ref<!fir.array<100x400xf32>>, i64, i64) -> !fir.ref<f32>
! CHECK:                 hlfir.yield %[[VAL_50]] : !fir.ref<f32>
! CHECK:               }
! CHECK:               hlfir.region_assign {
! CHECK:                 %[[VAL_51:.*]] = arith.constant 1.000000e+00 : f32
! CHECK:                 %[[VAL_52:.*]] = fir.load %[[VAL_23]] : !fir.ref<i32>
! CHECK:                 %[[VAL_53:.*]] = fir.convert %[[VAL_52]] : (i32) -> i64
! CHECK:                 %[[VAL_54:.*]] = fir.load %[[VAL_21]] : !fir.ref<i32>
! CHECK:                 %[[VAL_55:.*]] = fir.convert %[[VAL_54]] : (i32) -> i64
! CHECK:                 %[[VAL_56:.*]] = hlfir.designate %[[VAL_11]]#0 (%[[VAL_53]], %[[VAL_55]])  : (!fir.ref<!fir.array<200x200xf32>>, i64, i64) -> !fir.ref<f32>
! CHECK:                 %[[VAL_57:.*]] = fir.load %[[VAL_56]] : !fir.ref<f32>
! CHECK:                 %[[VAL_58:.*]] = arith.divf %[[VAL_51]], %[[VAL_57]] fastmath<contract> : f32
! CHECK:                 hlfir.yield %[[VAL_58]] : f32
! CHECK:               } to {
! CHECK:                 %[[VAL_59:.*]] = fir.load %[[VAL_21]] : !fir.ref<i32>
! CHECK:                 %[[VAL_60:.*]] = fir.convert %[[VAL_59]] : (i32) -> i64
! CHECK:                 %[[VAL_61:.*]] = arith.constant 200 : i32
! CHECK:                 %[[VAL_62:.*]] = fir.load %[[VAL_23]] : !fir.ref<i32>
! CHECK:                 %[[VAL_63:.*]] = arith.addi %[[VAL_61]], %[[VAL_62]] overflow<nsw> : i32
! CHECK:                 %[[VAL_64:.*]] = fir.convert %[[VAL_63]] : (i32) -> i64
! CHECK:                 %[[VAL_65:.*]] = hlfir.designate %[[VAL_7]]#0 (%[[VAL_60]], %[[VAL_64]])  : (!fir.ref<!fir.array<100x400xf32>>, i64, i64) -> !fir.ref<f32>
! CHECK:                 hlfir.yield %[[VAL_65]] : !fir.ref<f32>
! CHECK:               }
! CHECK:             }
! CHECK:           }
! CHECK:         }
! CHECK:         return
! CHECK:       }

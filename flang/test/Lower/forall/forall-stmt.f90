! Test forall lowering

! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

!*** Test a FORALL statement
subroutine test_forall_stmt(x, mask)

  logical :: mask(200)
  real :: x(200)
  forall (i=1:100,mask(i)) x(i) = 1.
end subroutine test_forall_stmt

! CHECK-LABEL: func.func @_QPtest_forall_stmt(
! CHECK-SAME:    %[[ARG0:.*]]: !fir.ref<!fir.array<200xf32>>{{.*}}, %[[ARG1:.*]]: !fir.ref<!fir.array<200x!fir.logical<4>>>{{.*}}) {
! CHECK:         %[[MASK:.*]]:2 = hlfir.declare %[[ARG1]]
! CHECK:         %[[X:.*]]:2 = hlfir.declare %[[ARG0]]
! CHECK:         hlfir.forall lb {
! CHECK:           hlfir.yield %{{.*}} : i32
! CHECK:         } ub {
! CHECK:           hlfir.yield %{{.*}} : i32
! CHECK:         }  (%[[I:.*]]: i32) {
! CHECK:           %[[IDX_I:.*]] = hlfir.forall_index "i" %[[I]]
! CHECK:           hlfir.forall_mask {
! CHECK:             %[[VAL_I:.*]] = fir.load %[[IDX_I]]
! CHECK:             %[[I_I64:.*]] = fir.convert %[[VAL_I]] : (i32) -> i64
! CHECK:             %[[MASK_I_REF:.*]] = hlfir.designate %[[MASK]]#0 (%[[I_I64]])
! CHECK:             %[[MASK_I:.*]] = fir.load %[[MASK_I_REF]]
! CHECK:             %[[COND:.*]] = fir.convert %[[MASK_I]] : (!fir.logical<4>) -> i1
! CHECK:             hlfir.yield %[[COND]] : i1
! CHECK:           } do {
! CHECK:             hlfir.region_assign {
! CHECK:               %[[CST:.*]] = arith.constant 1.000000e+00 : f32
! CHECK:               hlfir.yield %[[CST]] : f32
! CHECK:             } to {
! CHECK:               %[[VAL_I2:.*]] = fir.load %[[IDX_I]]
! CHECK:               %[[I_I64_2:.*]] = fir.convert %[[VAL_I2]] : (i32) -> i64
! CHECK:               %[[X_I_REF:.*]] = hlfir.designate %[[X]]#0 (%[[I_I64_2]])
! CHECK:               hlfir.yield %[[X_I_REF]] : !fir.ref<f32>
! CHECK:             }
! CHECK:           }
! CHECK:         }
! CHECK:         return
! CHECK:       }

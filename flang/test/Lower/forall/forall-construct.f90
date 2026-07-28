! Test forall lowering

! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

!*** Test a FORALL construct
subroutine test_forall_construct(a,b)
  integer :: i, j
  real :: a(:,:), b(:,:)
  forall (i=1:ubound(a,1), j=1:ubound(a,2), b(j,i) > 0.0)
     a(i,j) = b(j,i) / 3.14
  end forall
end subroutine test_forall_construct

! CHECK-LABEL: func.func @_QPtest_forall_construct(
! CHECK-SAME:     %[[ARG0:.*]]: !fir.box<!fir.array<?x?xf32>>{{.*}}, %[[ARG1:.*]]: !fir.box<!fir.array<?x?xf32>>{{.*}}) {
! CHECK:         %[[VAL_1:.*]]:2 = hlfir.declare %[[ARG0]]
! CHECK:         %[[VAL_2:.*]]:2 = hlfir.declare %[[ARG1]]
! CHECK:         hlfir.forall lb {
! CHECK:           hlfir.yield %{{.*}} : i32
! CHECK:         } ub {
! CHECK:           hlfir.yield %{{.*}} : i32
! CHECK:         }  (%[[I:.*]]: i32) {
! CHECK:           %[[IDX_I:.*]] = hlfir.forall_index "i" %[[I]]
! CHECK:           hlfir.forall lb {
! CHECK:             hlfir.yield %{{.*}} : i32
! CHECK:           } ub {
! CHECK:             hlfir.yield %{{.*}} : i32
! CHECK:           }  (%[[J:.*]]: i32) {
! CHECK:             %[[IDX_J:.*]] = hlfir.forall_index "j" %[[J]]
! CHECK:             hlfir.forall_mask {
! CHECK:               %[[VAL_J:.*]] = fir.load %[[IDX_J]]
! CHECK:               %[[VAL_I:.*]] = fir.load %[[IDX_I]]
! CHECK:               %[[B_JI_REF:.*]] = hlfir.designate %[[VAL_2]]#0 (%{{.*}}, %{{.*}})
! CHECK:               %[[B_JI:.*]] = fir.load %[[B_JI_REF]]
! CHECK:               %[[COND:.*]] = arith.cmpf ogt, %[[B_JI]], %{{.*}}
! CHECK:               hlfir.yield %[[COND]] : i1
! CHECK:             } do {
! CHECK:               hlfir.region_assign {
! CHECK:                 %[[B_JI_REF2:.*]] = hlfir.designate %[[VAL_2]]#0 (%{{.*}}, %{{.*}})
! CHECK:                 %[[B_JI2:.*]] = fir.load %[[B_JI_REF2]]
! CHECK:                 %[[RES:.*]] = arith.divf %[[B_JI2]], %{{.*}}
! CHECK:                 hlfir.yield %[[RES]] : f32
! CHECK:               } to {
! CHECK:                 %[[A_IJ_REF:.*]] = hlfir.designate %[[VAL_1]]#0 (%{{.*}}, %{{.*}})
! CHECK:                 hlfir.yield %[[A_IJ_REF]] : !fir.ref<f32>
! CHECK:               }
! CHECK:             }
! CHECK:           }
! CHECK:         }
! CHECK:         return
! CHECK:       }

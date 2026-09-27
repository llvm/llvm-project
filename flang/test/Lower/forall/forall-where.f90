! Test forall lowering

! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

!*** Test a FORALL construct with a nested WHERE construct.
!    This has both an explicit and implicit iteration space. The WHERE construct
!    makes the assignments conditional and the where mask evaluation must happen
!    prior to evaluating the array assignment statement.
subroutine test_nested_forall_where(a,b)
  type t
     real data(100)
  end type t
  type(t) :: a(:,:), b(:,:)
  forall (i=1:ubound(a,1), j=1:ubound(a,2))
     where (b(j,i)%data > 0.0)
        a(i,j)%data = b(j,i)%data / 3.14
     elsewhere
        a(i,j)%data = -b(j,i)%data
     end where
  end forall
end subroutine test_nested_forall_where

! CHECK-LABEL: func.func @_QPtest_nested_forall_where(
! CHECK: hlfir.forall
! CHECK: (%[[arg2:.*]]: i32) {
! CHECK:   %[[i:.*]] = hlfir.forall_index "i" %[[arg2]] : (i32) -> !fir.ref<i32>
! CHECK:   hlfir.forall
! CHECK:   (%[[arg3:.*]]: i32) {
! CHECK:     %[[j:.*]] = hlfir.forall_index "j" %[[arg3]] : (i32) -> !fir.ref<i32>
! CHECK:     hlfir.where {
! CHECK:       %[[mask:.*]] = hlfir.elemental {{.*}} {
! CHECK:         arith.cmpf ogt, {{.*}}
! CHECK:       }
! CHECK:       hlfir.yield %[[mask]]
! CHECK:     } do {
! CHECK:       hlfir.region_assign {
! CHECK:         %[[res:.*]] = hlfir.elemental {{.*}} {
! CHECK:           arith.divf {{.*}}
! CHECK:         }
! CHECK:         hlfir.yield %[[res]]
! CHECK:       } to {
! CHECK:       }
! CHECK:     hlfir.elsewhere do {
! CHECK:       hlfir.region_assign {
! CHECK:         %[[res:.*]] = hlfir.elemental {{.*}} {
! CHECK:           arith.negf {{.*}}
! CHECK:         }
! CHECK:         hlfir.yield %[[res]]
! CHECK:       } to {
! CHECK:       }
! CHECK:     }
! CHECK:   }
! CHECK: }

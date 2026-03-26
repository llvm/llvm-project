! Test forall lowering

! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

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

! CHECK-LABEL: func.func @_QPtest9(
! CHECK: hlfir.forall
! CHECK: (%[[arg3:.*]]: i32) {
! CHECK:   %[[i:.*]] = hlfir.forall_index "i" %[[arg3]] : (i32) -> !fir.ref<i32>
! CHECK:   hlfir.region_assign {
! CHECK:     %[[i_val:.*]] = fir.load %[[i]]
! CHECK:     %[[i_idx:.*]] = fir.convert %[[i_val]] : (i32) -> i64
! CHECK:     %[[a_i_ref:.*]] = hlfir.designate {{.*}} (%[[i_idx]])
! CHECK:     %[[a_i:.*]] = fir.load %[[a_i_ref]]
! CHECK:     %[[i_val2:.*]] = fir.load %[[i]]
! CHECK:     %[[i_idx2:.*]] = fir.convert %[[i_val2]] : (i32) -> i64
! CHECK:     %[[b_i_ref:.*]] = hlfir.designate {{.*}} (%[[i_idx2]])
! CHECK:     %[[b_i:.*]] = fir.load %[[b_i_ref]]
! CHECK:     %[[res:.*]] = arith.addf %[[a_i]], %[[b_i]]
! CHECK:     hlfir.yield %[[res]]
! CHECK:   } to {
! CHECK:     %[[i_val3:.*]] = fir.load %[[i]]
! CHECK:     %[[i_plus_1:.*]] = arith.addi %[[i_val3]], {{.*}}
! CHECK:     %[[i_plus_1_idx:.*]] = fir.convert %[[i_plus_1]] : (i32) -> i64
! CHECK:     %[[a_i_plus_1_ref:.*]] = hlfir.designate {{.*}} (%[[i_plus_1_idx]])
! CHECK:     hlfir.yield %[[a_i_plus_1_ref]]
! CHECK:   }
! CHECK: }

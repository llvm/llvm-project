! RUN: bbc -fopenacc -emit-hlfir %s -o - | FileCheck %s

subroutine copy_then_reduction()
  integer :: x
  !$acc parallel copy(x) reduction(+:x)
  x = x + 1
  !$acc end parallel
end subroutine

! CHECK-LABEL: func.func @_QPcopy_then_reduction()
! CHECK: %[[COPY1:.*]] = acc.copyin varPtr({{.*}}) dataClause(acc_copy) name("x") -> !fir.ref<i32>
! CHECK: %[[REDUCTION1:.*]] = acc.reduction varPtr({{.*}}) recipe({{.*}}) name("x") -> !fir.ref<i32>
! CHECK: acc.parallel dataOperands(%[[COPY1]] : !fir.ref<i32>) reduction(%[[REDUCTION1]] : !fir.ref<i32>) {
! TODO: The region body binds x to the first mapping, so lowering depends on
! clause order. The reduction mapping should take priority in both cases. The
! two checks below record the current, wrong behavior. When it is fixed they
! will fail: change them to expect %[[REDUCTION1]], as in reduction_then_copy
! below, rather than deleting them.
! CHECK: hlfir.declare %[[COPY1]]
! CHECK: acc.copyout accPtr(%[[COPY1]] : !fir.ref<i32>)

subroutine reduction_then_copy()
  integer :: x
  !$acc parallel reduction(+:x) copy(x)
  x = x + 1
  !$acc end parallel
end subroutine

! CHECK-LABEL: func.func @_QPreduction_then_copy()
! CHECK: %[[REDUCTION2:.*]] = acc.reduction varPtr({{.*}}) recipe({{.*}}) name("x") -> !fir.ref<i32>
! CHECK: %[[COPY2:.*]] = acc.copyin varPtr({{.*}}) dataClause(acc_copy) name("x") -> !fir.ref<i32>
! CHECK: acc.parallel dataOperands(%[[COPY2]] : !fir.ref<i32>) reduction(%[[REDUCTION2]] : !fir.ref<i32>) {
! CHECK: hlfir.declare %[[REDUCTION2]]
! CHECK: acc.copyout accPtr(%[[COPY2]] : !fir.ref<i32>)

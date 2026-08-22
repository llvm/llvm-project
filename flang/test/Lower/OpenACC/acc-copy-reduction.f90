! RUN: bbc -fopenacc -emit-hlfir %s -o - | FileCheck %s

subroutine copy_then_reduction()
  integer :: x
  !$acc parallel copy(x) reduction(+:x)
  x = x + 1
  !$acc end parallel
end subroutine

! CHECK-LABEL: func.func @_QPcopy_then_reduction()
! CHECK: acc.reduction varPtr({{.*}}) recipe({{.*}}) name("x") -> !fir.ref<i32>
! CHECK-NOT: acc.copy
! CHECK: acc.parallel reduction({{.*}}) {

subroutine reduction_then_copy()
  integer :: x
  !$acc parallel reduction(+:x) copy(x)
  x = x + 1
  !$acc end parallel
end subroutine

! CHECK-LABEL: func.func @_QPreduction_then_copy()
! CHECK: acc.reduction varPtr({{.*}}) recipe({{.*}}) name("x") -> !fir.ref<i32>
! CHECK-NOT: acc.copy
! CHECK: acc.parallel reduction({{.*}}) {

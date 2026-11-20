! A VOLATILE EQUIVALENCE member lowers to a fir.ptr without the volatile
! qualifier. The reduction must still be treated as volatile: named
! @add_reduction_volatile_i32, with no atomic region.

! RUN: %flang_fc1 -emit-hlfir -fopenmp -o - %s | FileCheck %s

! CHECK:       omp.declare_reduction @add_reduction_volatile_i32 : i32
! CHECK-NOT:     llvm.atomicrmw
! CHECK-LABEL: func.func @_QPveq
! CHECK:         omp.wsloop {{.*}}reduction(@add_reduction_volatile_i32 %{{.*}} -> %{{.*}} : !fir.ref<i32, volatile>)

subroutine veq(n)
  integer :: n, i
  integer, volatile :: vi
  real :: x
  equivalence (vi, x)
  vi = 0
  !$omp parallel do reduction(+:vi)
  do i = 1, n
    vi = vi + 1
  end do
end subroutine

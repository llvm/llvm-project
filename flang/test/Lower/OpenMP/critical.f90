! REQUIRES: openmp_runtime

!RUN: %flang_fc1 -emit-hlfir %openmp_flags %s -o - | FileCheck %s

!CHECK: omp.critical.declare @help2
!CHECK: omp.critical.declare @help1 hint(contended)

subroutine omp_critical()
  use omp_lib
  integer :: x, y
!CHECK: omp.critical(@help1)
!$OMP CRITICAL(help1) HINT(omp_lock_hint_contended)
  x = x + y
!CHECK: omp.terminator
!$OMP END CRITICAL(help1)

! Test that the same name can be used again
! Also test with the zero hint expression
!CHECK: omp.critical(@help2)
!$OMP CRITICAL(help2) HINT(omp_lock_hint_none)
  x = x - y
!CHECK: omp.terminator
!$OMP END CRITICAL(help2)

!CHECK: omp.critical
!$OMP CRITICAL
  y = x + y
!CHECK: omp.terminator
!$OMP END CRITICAL
end subroutine omp_critical


! Tests that privatization for pre-determined variables (here `i`) is properly
! handled.
subroutine predetermined_privatization()
  integer :: a(10), i

  !CHECK: omp.parallel
  !$omp parallel do

  !CHECK: %[[PRIV_I_ALLOC:.*]] = fir.alloca i32 {bindc_name = "i", pinned, {{.*}}}
  !CHECK: %[[PRIV_I_DECL:.*]]:2 = hlfir.declare %[[PRIV_I_ALLOC]]
  do i = 2, 10
    !CHECK: omp.wsloop
    !CHECK: omp.loop_nest (%[[IV:[^[:space:]]+]])
    !CHECK: fir.store %[[IV]] to %[[PRIV_I_DECL]]#1
    !CHECK: omp.critical
    !$omp critical
    a(i) = a(i-1) + 1
    !$omp end critical
  end do
  !$omp end parallel do
end

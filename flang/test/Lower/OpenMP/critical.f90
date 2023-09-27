!RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s

!CHECK: omp.critical.declare @help2 hint(none)
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

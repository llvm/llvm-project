! RUN: %flang_fc1 -fopenmp -fsyntax-only %s 2>&1 | FileCheck %s --allow-empty
! CHECK-NOT: error:
! CHECK-NOT: warning:

subroutine sub1()
  common /com/ j
  j = 10
!$omp parallel do firstprivate(j) lastprivate(/com/)
  do i = 1, 10
     j = j + 1
  end do
!$omp end parallel do
end

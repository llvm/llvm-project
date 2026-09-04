! RUN: %flang_fc1 -fopenmp -fopenmp-version=51 -fsyntax-only %s 2>&1 | FileCheck %s --allow-empty
! CHECK-NOT: error:

! Regression test for issue #162033.
! Verify that a named COMMON block can appear in a data-sharing clause together
! with one of its members in another clause on the same construct. This is valid
! in OpenMP >= 5.1 because:
!  - A named COMMON in a clause is equivalent to listing all its explicit members
!  - A list item may appear in both FIRSTPRIVATE and LASTPRIVATE on the same directive
! OpenMP 5.0 explicitly forbade this combination.

subroutine sub1()
  common /com/ j
  j = 10
!$omp parallel do firstprivate(j) lastprivate(/com/)
  do i = 1, 10
     j = j + 1
  end do
!$omp end parallel do
end

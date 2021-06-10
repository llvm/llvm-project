! RUN: %S/test_errors.sh %s %t %flang_fc1 -fopenmp
! REQUIRES: shell
! OpenMP Version 4.5
! 2.15.4.2 copyprivate Clause
! Pointers with the INTENT(IN) attribute may not appear in a copyprivate clause.

subroutine omp_copyprivate(p)
  integer :: a(10), b(10), c(10)
  integer, pointer, intent(in) :: p

  a = 10
  b = 20

  !$omp parallel
  !$omp single
  c = a + b + p
  !ERROR: COPYPRIVATE variable 'p' is not PRIVATE or THREADPRIVATE in outer context
  !ERROR: Pointer 'p' with the INTENT(IN) attribute may not appear in a COPYPRIVATE clause
  !$omp end single copyprivate(p)
  !$omp end parallel

  print *, c

end subroutine omp_copyprivate

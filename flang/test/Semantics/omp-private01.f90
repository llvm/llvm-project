! RUN: %S/test_errors.sh %s %t %flang -fopenmp
! REQUIRES: shell
! OpenMP Version 4.5
! 2.15.3.3 private Clause
! Pointers with the INTENT(IN) attribute may not appear in a private clause.

subroutine omp_private(p)
  integer :: a(10), b(10), c(10)
  integer, pointer, intent(in) :: p

  a = 10
  b = 20

  !ERROR: Pointer 'p' with the INTENT(IN) attribute may not appear in a PRIVATE clause
  !$omp parallel private(p)
  c = a + b + p
  !$omp end parallel

  print *, c

end subroutine omp_private

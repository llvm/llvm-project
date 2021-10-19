! RUN: %python %S/test_errors.py %s %flang_fc1 -fopenmp
! OpenMP Version 4.5
! 2.15.3.5 lastprivate Clause
! A variable that appears in a lastprivate clause must be definable.

module protected_var
  integer, protected :: p
end module protected_var

program omp_lastprivate
  use protected_var
  integer :: i, a(10), b(10), c(10)
  integer, parameter :: k = 10

  a = 10
  b = 20

  !ERROR: Variable 'k' on the LASTPRIVATE clause is not definable
  !$omp parallel do lastprivate(k)
  do i = 1, 10
    c(i) = a(i) + b(i) + k
  end do
  !$omp end parallel do

  !ERROR: Variable 'p' on the LASTPRIVATE clause is not definable
  !$omp parallel do lastprivate(p)
  do i = 1, 10
    c(i) = a(i) + b(i) + k
  end do
  !$omp end parallel do

  call omp_lastprivate_sb(i)

  print *, c

end program omp_lastprivate

subroutine omp_lastprivate_sb(m)
  integer :: i, a(10), b(10), c(10)
  integer, intent(in) :: m

  a = 10
  b = 20

  !ERROR: Variable 'm' on the LASTPRIVATE clause is not definable
  !$omp parallel do lastprivate(m)
  do i = 1, 10
    c(i) = a(i) + b(i) + m
  end do
  !$omp end parallel do

  print *, c

end subroutine omp_lastprivate_sb

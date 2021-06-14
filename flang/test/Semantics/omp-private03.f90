! RUN: %S/test_errors.sh %s %t %flang -fopenmp
! REQUIRES: shell
! OpenMP Version 4.5
! Variables that appear in expressions for statement function definitions
! may not appear in private, firstprivate or lastprivate clauses.

subroutine stmt_function(temp)

  integer :: i, p, q, r
  real :: c, f, s, v, t(10)
  real, intent(in) :: temp

  c(temp) = p * (temp - q) / r
  f(temp) = q + (temp * r/p)
  v(temp) = c(temp) + f(temp)/2 - s

  p = 5
  q = 32
  r = 9

  !ERROR: Variable 'p' in STATEMENT FUNCTION expression cannot be in a PRIVATE clause
  !$omp parallel private(p)
  s = c(temp)
  !$omp end parallel

  !ERROR: Variable 's' in STATEMENT FUNCTION expression cannot be in a FIRSTPRIVATE clause
  !$omp parallel firstprivate(s)
  s = s + f(temp)
  !$omp end parallel

  !ERROR: Variable 's' in STATEMENT FUNCTION expression cannot be in a LASTPRIVATE clause
  !$omp parallel do lastprivate(s, t)
  do i = 1, 10
    t(i) = v(temp) + i - s
  end do
  !$omp end parallel do

  print *, t

end subroutine stmt_function

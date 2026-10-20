!RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=52

! The CONDITIONAL lastprivate modifier must not be specified on a distribute
! directive, including combined and composite forms that have distribute as a
! leaf construct.

subroutine foo(n)
  integer :: n, x, i
  x = 1
  !$omp teams
!ERROR: 'CONDITIONAL' modifier on lastprivate clause with DISTRIBUTE directive is not allowed
  !$omp distribute lastprivate(conditional: x)
  do i = 1, n
    if (mod(i, 2) == 0) x = i
  enddo
  !$omp end distribute
  !$omp end teams
end subroutine

subroutine bar(n)
  integer :: n, x, i
  x = 1
!ERROR: 'CONDITIONAL' modifier on lastprivate clause with DISTRIBUTE directive is not allowed
  !$omp teams distribute parallel do lastprivate(conditional: x)
  do i = 1, n
    if (mod(i, 2) == 0) x = i
  enddo
end subroutine

! A composite form that has a worksharing-loop/SIMD leaf is still rejected: the
! restriction is applied conservatively to any construct with a distribute leaf.
subroutine baz(n)
  integer :: n, x, i
  x = 1
  !$omp teams
!ERROR: 'CONDITIONAL' modifier on lastprivate clause with DISTRIBUTE directive is not allowed
  !$omp distribute parallel do simd lastprivate(conditional: x)
  do i = 1, n
    if (mod(i, 2) == 0) x = i
  enddo
  !$omp end distribute parallel do simd
  !$omp end teams
end subroutine

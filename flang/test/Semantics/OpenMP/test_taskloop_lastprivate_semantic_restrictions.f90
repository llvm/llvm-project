!RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=52

! The CONDITIONAL lastprivate modifier must not be specified on a taskloop
! directive, including combined and composite forms that have taskloop as a
! leaf construct.

subroutine foo()
  integer :: x, i
  x = 1
!ERROR: 'CONDITIONAL' modifier on lastprivate clause with TASKLOOP directive is not allowed
  !$omp taskloop lastprivate(conditional: x)
  do i = 1, 100
    if (mod(i, 2) == 0) x = i
  enddo
  !$omp end taskloop
end

! Composite: taskloop simd has taskloop as a leaf construct.
subroutine foo_simd()
  integer :: x, i
  x = 1
!ERROR: 'CONDITIONAL' modifier on lastprivate clause with TASKLOOP directive is not allowed
  !$omp taskloop simd lastprivate(conditional: x)
  do i = 1, 100
    if (mod(i, 2) == 0) x = i
  enddo
  !$omp end taskloop simd
end

! Combined: masked taskloop.
subroutine masked_tl()
  integer :: x, i
  x = 1
!ERROR: 'CONDITIONAL' modifier on lastprivate clause with TASKLOOP directive is not allowed
  !$omp masked taskloop lastprivate(conditional: x)
  do i = 1, 100
    if (mod(i, 2) == 0) x = i
  enddo
  !$omp end masked taskloop
end

! Combined/composite: masked taskloop simd.
subroutine masked_tl_simd()
  integer :: x, i
  x = 1
!ERROR: 'CONDITIONAL' modifier on lastprivate clause with TASKLOOP directive is not allowed
  !$omp masked taskloop simd lastprivate(conditional: x)
  do i = 1, 100
    if (mod(i, 2) == 0) x = i
  enddo
  !$omp end masked taskloop simd
end

! Combined: parallel masked taskloop.
subroutine par_masked_tl()
  integer :: x, i
  x = 1
!ERROR: 'CONDITIONAL' modifier on lastprivate clause with TASKLOOP directive is not allowed
  !$omp parallel masked taskloop lastprivate(conditional: x)
  do i = 1, 100
    if (mod(i, 2) == 0) x = i
  enddo
  !$omp end parallel masked taskloop
end

! Combined/composite: parallel masked taskloop simd.
subroutine par_masked_tl_simd()
  integer :: x, i
  x = 1
!ERROR: 'CONDITIONAL' modifier on lastprivate clause with TASKLOOP directive is not allowed
  !$omp parallel masked taskloop simd lastprivate(conditional: x)
  do i = 1, 100
    if (mod(i, 2) == 0) x = i
  enddo
  !$omp end parallel masked taskloop simd
end

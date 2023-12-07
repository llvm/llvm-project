! RUN: %python %S/../test_errors.py %s %flang -fopenmp -Werror
! OpenMP Version 5.0
! 2.19.4.4 firstprivate Clause
! 2.19.4.5 lastprivate Clause
! 2.19.6.1 copyin Clause
! 2.19.6.2 copyprivate Clause
! If the list item is a polymorphic variable with the allocatable attribute,
! the behavior is unspecified.

subroutine firstprivate()
  class(*), allocatable, save :: x

  !PORTABILITY: If a polymorphic variable with allocatable attribute 'x' is in FIRSTPRIVATE clause, the behavior is unspecified
  !$omp parallel firstprivate(x)
    call sub()
  !$omp end parallel

end

subroutine lastprivate()
  class(*), allocatable, save :: x

  !PORTABILITY: If a polymorphic variable with allocatable attribute 'x' is in LASTPRIVATE clause, the behavior is unspecified
  !$omp do lastprivate(x)
  do i = 1, 10
    call sub()
  enddo
  !$omp end do

end

subroutine copyin()
  class(*), allocatable, save :: x
  !$omp threadprivate(x)

  !PORTABILITY: If a polymorphic variable with allocatable attribute 'x' is in COPYIN clause, the behavior is unspecified
  !$omp parallel copyin(x)
    call sub()
  !$omp end parallel

end

subroutine copyprivate()
  class(*), allocatable, save :: x
  !$omp threadprivate(x)

  !PORTABILITY: If a polymorphic variable with allocatable attribute 'x' is in COPYPRIVATE clause, the behavior is unspecified
  !$omp single copyprivate(x)
    call sub()
  !$omp end single

end

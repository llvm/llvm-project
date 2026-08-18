! RUN: %flang_fc1 -fopenmp -fsyntax-only %s

! Test that various ways of nesting combined and standalone `loop` constructs
! inside of `teams` are accepted.

subroutine test()
  implicit none
  integer :: i

  !$omp teams
  !$omp parallel loop
  do i=1, 10
    call foo()
  end do
  !$omp end teams

  !$omp target teams
  !$omp parallel loop
  do i=1, 10
    call foo()
  end do
  !$omp end target teams

  !$omp target teams
  !$omp loop
  do i=1, 10
    call foo()
  end do
  !$omp end target teams

  !$omp teams
  !$omp loop
  do i=1, 10
    call foo()
  end do
  !$omp end teams
end subroutine

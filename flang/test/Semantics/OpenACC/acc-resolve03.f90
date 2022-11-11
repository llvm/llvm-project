! RUN: %flang_fc1 -fopenacc %s
! A regression test to check that
! arbitrary compiler directives do not generate errors
! inside OpenACC collapsed loops
subroutine foo
  integer, parameter :: loop_bound = 42
  integer :: a
  integer :: b
  integer :: c

  !$acc parallel
  do a = 0, loop_bound
    !$acc loop collapse(2)
    do b = 0, loop_bound
      !dir$ ivdep
      do c = 0, loop_bound
      enddo
    enddo
  enddo
  !$acc end parallel
end subroutine foo

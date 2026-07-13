! RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenacc

subroutine test_sentinel()
! Test for error since we currently do not have an OpenACC module upstream.
!ERROR: Cannot parse module file for module 'openacc': Source file 'openacc.mod' was not found
  !@acc use openacc
  integer :: i

  !$acc parallel loop
  do i = 1, 10
  end do
  !$acc end parallel

end subroutine

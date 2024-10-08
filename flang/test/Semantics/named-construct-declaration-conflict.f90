!RUN: %python %S/test_errors.py %s %flang_fc1

subroutine foo()
  integer :: nxloop
!ERROR: 'nxloop' is already declared in this scoping unit
  nxloop: do i = 1, 100
  enddo nxloop
end subroutine



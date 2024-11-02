!RUN: %python %S/test_errors.py %s %flang_fc1

subroutine foo()
  integer :: xyz
!ERROR: 'xyz' is already declared in this scoping unit
  xyz: do i = 1, 100
  enddo xyz
end subroutine



! RUN: %python %S/test_errors.py %s %flang_fc1
! Check for C1546 and 18.3.6

! ERROR: 'test1' may not have both the BIND(C) and ELEMENTAL attributes
elemental subroutine test1() bind(c)
end

subroutine test3(x) bind(c)
  ! ERROR: VALUE attribute may not apply to an array in a BIND(C) procedure
  integer, value :: x(100)
end

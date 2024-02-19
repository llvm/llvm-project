! RUN: %python %S/test_modfile.py %s %flang_fc1
! Named constant array with non-default lower bound
module m
  real, parameter :: x(0:0) = [0.]
end

!Expect: m.mod
!module m
!real(4),parameter::x(0_8:0_8)=[REAL(4)::0._4]
!end

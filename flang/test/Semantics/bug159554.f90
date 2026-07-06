!RUN: %python %S/test_errors.py %s %flang_fc1
use, intrinsic :: iso_c_binding
interface c_funloc
!ERROR: 'c_funloc' is already declared in this scoping unit
  function c_funloc()
  end function
end interface
end

!RUN: %python %S/test_errors.py %s %flang_fc1
use, intrinsic :: iso_c_binding
!WARNING: 'c_funloc' should not be the name of both a generic interface and a procedure unless it is a specific procedure of the generic [-Whomonymous-specific]
interface c_funloc
!ERROR: 'c_funloc' is already declared in this scoping unit
  function c_funloc()
  end function
end interface
end

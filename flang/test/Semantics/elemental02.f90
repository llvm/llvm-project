! RUN: %python %S/test_errors.py %s %flang_fc1
subroutine s(a)
  real a(*)
  interface
    elemental function ef(efarg)
      real, intent(in) :: efarg
    end
  end interface
!ERROR: Whole assumed-size array 'a' may not be used as an argument to an elemental procedure
  print *, sqrt(a)
!ERROR: Whole assumed-size array 'a' may not be used as an argument to an elemental procedure
  print *, ef(a)
end

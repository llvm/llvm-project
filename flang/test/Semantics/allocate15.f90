! RUN: %python %S/test_errors.py %s %flang_fc1
! Do not crash when ALLOCATE has SOURCE= with a constant after module USE cycle errors.
!ERROR: Some modules in this compilation unit form one or more cycles of dependence
module m1
  use m3
end
module m3
  use m1
end
subroutine s()
  allocate(x1, source = [1, 2, 3, 4, 5])
end

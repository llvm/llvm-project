! RUN: %python %S/test_folding.py %s %flang_fc1
! Test folding of character array conversion.
module m
  character(*,4), parameter :: str4arr(1) = ['a']
  logical, parameter :: test = str4arr(1) == 4_'a'
end

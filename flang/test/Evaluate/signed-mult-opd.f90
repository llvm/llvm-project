! RUN: %python %S/test_folding.py %s %flang_fc1
module m
  integer, parameter :: j = 2
  ! standard cases
  logical, parameter :: test_1 = -j**2 == -4
  logical, parameter :: test_2 = 4-j**2 == 0
  ! extension cases
  logical, parameter :: test_3 = 4+-j**2 == 0 ! not 8
  logical, parameter :: test_4 = 2*-j**2 == -8 ! not 8
  logical, parameter :: test_5 = -j**2+-j**2 == -8 ! not 8
  logical, parameter :: test_6 = j**2*-j**2 == -16 ! not 16
end

! RUN: %python %S/test_folding.py %s %flang_fc1
! Tests folding of SCALE()
module m
  logical, parameter :: test_1 = scale(1.0, 1) == 2.0
  logical, parameter :: test_2 = scale(0.0, 1) == 0.0
  logical, parameter :: test_3 = sign(1.0, scale(-0.0, 1)) == -1.0
  logical, parameter :: test_4 = sign(1.0, scale(0.0, 0)) == 1.0
  logical, parameter :: test_5 = scale(1.0, -1) == 0.5
  logical, parameter :: test_6 = scale(2.0, -1) == 1.0
end module


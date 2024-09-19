! RUN: %python %S/test_folding.py %s %flang_fc1
! Tests folding of SCALE()
module m
  logical, parameter :: test_1 = scale(1.0, 1) == 2.0
  logical, parameter :: test_2 = scale(0.0, 1) == 0.0
  logical, parameter :: test_3 = sign(1.0, scale(-0.0, 1)) == -1.0
  logical, parameter :: test_4 = sign(1.0, scale(0.0, 0)) == 1.0
  logical, parameter :: test_5 = scale(1.0, -1) == 0.5
  logical, parameter :: test_6 = scale(2.0, -1) == 1.0
  logical, parameter :: test_7 = scale(huge(0.d0), -1200) == 1.0440487148797638d-53
  logical, parameter :: test_8 = scale(tiny(0.d0), 1200) == 3.8312388521647221d053
end module

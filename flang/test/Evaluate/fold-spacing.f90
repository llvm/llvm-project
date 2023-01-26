! RUN: %python %S/test_folding.py %s %flang_fc1
! Tests folding of SPACING() and RRSPACING
module m
  logical, parameter :: test_1 = spacing(3.0) == scale(1.0, -22)
  logical, parameter :: test_2 = spacing(-3.0) == scale(1.0, -22)
  logical, parameter :: test_3 = spacing(3.0d0) == scale(1.0, -51)
  logical, parameter :: test_4 = spacing(0.) == tiny(0.)
  logical, parameter :: test_5 = spacing(tiny(0.)) == 1.e-45
  logical, parameter :: test_6 = spacing(8388608.) == 1.
  logical, parameter :: test_11 = rrspacing(3.0) == scale(0.75, 24)
  logical, parameter :: test_12 = rrspacing(-3.0) == scale(0.75, 24)
  logical, parameter :: test_13 = rrspacing(3.0d0) == scale(0.75, 53)
  logical, parameter :: test_14 = rrspacing(0.) == 0.
end module

! RUN: %python %S/test_folding.py %s %flang_fc1
module m
  complex, parameter :: z1 = 1. + (2., 3.)
  logical, parameter :: test_z1 = z1 == (3., 3.)
  complex, parameter :: z2 = 1 + (2., 3.)
  logical, parameter :: test_z2 = z2 == (3., 3.)
  complex, parameter :: z3 = 2. * (3., 4.)
  logical, parameter :: test_z3 = z3 == (6., 8.)
  complex, parameter :: z4 = 2 * (3., 4.)
  logical, parameter :: test_z4 = z4 == (6., 8.)
  complex, parameter :: z5 = 5. - (3., 4.)
  logical, parameter :: test_z5 = z5 == (2., -4.)
  complex, parameter :: z6 = 5 - (3., 4.)
  logical, parameter :: test_z6 = z6 == (2., -4.)
  complex, parameter :: z11 = (2., 3.) + 1.
  logical, parameter :: test_z11 = z11 == (3., 3.)
  complex, parameter :: z12 = (2., 3.) + 1
  logical, parameter :: test_z12 = z12 == (3., 3.)
  complex, parameter :: z13 = (3., 4.) * 2.
  logical, parameter :: test_z13 = z13 == (6., 8.)
  complex, parameter :: z14 = (3., 4.) * 2
  logical, parameter :: test_z14 = z14 == (6., 8.)
  complex, parameter :: z15 = (3., 4.) - 1.
  logical, parameter :: test_z15 = z15 == (2., 4.)
  complex, parameter :: z16 = (3., 4.) - 1
  logical, parameter :: test_z16 = z16 == (2., 4.)
  complex, parameter :: z17 = (3., 4.) / 2.
  logical, parameter :: test_z17 = z17 == (1.5, 2.)
  complex, parameter :: z18 = (3., 4.) / 2
  logical, parameter :: test_z18 = z18 == (1.5, 2.)
end module

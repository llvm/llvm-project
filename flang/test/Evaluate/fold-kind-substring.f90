! RUN: %python %S/test_folding.py %s %flang_fc1
! Regression tests: a substring of a character literal has the kind of the
! literal (F2023 9.4.1), and its length folds from the substring bounds.
! Only type inquiries are asserted here: KIND() and LEN() do not require
! their argument to be a constant, and the value of a substring of a
! non-default-kind literal does not currently fold.
module m
  logical, parameter :: test_k4 = kind(4_"abcd"(2:3)) == 4
  logical, parameter :: test_k2 = kind(2_"ab"(1:1)) == 2
  logical, parameter :: test_k1 = kind(1_"ab"(1:1)) == 1
  logical, parameter :: test_l4 = len(4_"abcd"(2:3)) == 2
  logical, parameter :: test_l4a = len(4_"abcd"(:3)) == 3
  logical, parameter :: test_l4b = len(4_"abcd"(2:)) == 3
end module

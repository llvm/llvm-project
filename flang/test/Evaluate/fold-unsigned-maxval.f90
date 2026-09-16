! RUN: %python %S/test_folding.py %s %flang_fc1 -funsigned
! Tests MAXVAL/MINVAL folding
module repro
  logical, parameter :: max_ok = maxval([1u_1, 2u_1]) == 2u_1
  logical, parameter :: min_ok = minval([1u_1, 2u_1]) == 1u_1
end module

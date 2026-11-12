! Tests folding of SELECTED_REAL_KIND

! RUN: %python %S/test_folding.py %s %flang_fc1 -fdisable-real-16

module m
  logical, parameter :: test_16 = selected_real_kind(p=33) == -1
end

! RUN: %python %S/test_folding.py %s %flang_fc1
module m
  logical, parameter :: test_0  = selected_logical_kind( 0) == 1
  logical, parameter :: test_1  = selected_logical_kind( 1) == 1
  logical, parameter :: test_7  = selected_logical_kind( 7) == 1
  logical, parameter :: test_8  = selected_logical_kind( 8) == 1
  logical, parameter :: test_9  = selected_logical_kind( 9) == 2
  logical, parameter :: test_15 = selected_logical_kind(15) == 2
  logical, parameter :: test_16 = selected_logical_kind(16) == 2
  logical, parameter :: test_17 = selected_logical_kind(17) == 4
  logical, parameter :: test_31 = selected_logical_kind(31) == 4
  logical, parameter :: test_32 = selected_logical_kind(32) == 4
  logical, parameter :: test_33 = selected_logical_kind(33) == 8
  logical, parameter :: test_63 = selected_logical_kind(63) == 8
  logical, parameter :: test_64 = selected_logical_kind(64) == 8
  logical, parameter :: test_65 = selected_logical_kind(65) == -1
end

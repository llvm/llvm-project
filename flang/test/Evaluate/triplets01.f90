! RUN: %python %S/test_folding.py %s %flang_fc1
module m
  logical, parameter :: test01 = all([1:10:2] == [(j, j=1,10,2)])
  logical, parameter :: test02 = kind([1:20:2]) == kind(1)
  logical, parameter :: test03 = all([10:1:-3,123] == [(j, j=10,1,-3),123])
  logical, parameter :: test04 = kind([10:1:-3,123]) == kind(1)
  logical, parameter :: test05 = kind([10_2:1_2:-3_2,123_2]) == 2
  logical, parameter :: test06 = all([10_2:1_2:-3_2,123_2] == [(j, integer(2)::j=10,1,-3),123_2])
  logical, parameter :: test07 = kind([10_2:1_4:-3_2]) == 4
  logical, parameter :: test08 = kind([10_2:1_4]) == 4
end

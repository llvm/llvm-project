! RUN: %python %S/test_folding.py %s %flang_fc1
! Test folding of substrings
module m
  logical, parameter :: test_01a = "abc"(1:3) == "abc"
  logical, parameter :: test_01b = len("abc"(1:3)) == 3
  logical, parameter :: test_02a = "abc"(-1:-2) == ""
  logical, parameter :: test_02b = len("abc"(-1:-2)) == 0
  logical, parameter :: test_03a = "abc"(9999:4) == ""
  logical, parameter :: test_03b = len("abc"(9999:4)) == 0
  character(4), parameter :: ca(3) = ["abcd", "efgh", "ijkl"]
  logical, parameter :: test_04a = ca(2)(2:4) == "fgh"
  logical, parameter :: test_04b = len(ca(2)(2:4)) == 3
  logical, parameter :: test_05a = all(ca(:)(2:4) == ["bcd", "fgh", "jkl"])
  logical, parameter :: test_05b = len(ca(:)(2:4)) == 3
  logical, parameter :: test_06a = ca(1)(1:2)//ca(2)(2:3)//ca(3)(3:4) == "abfgkl"
  logical, parameter :: test_06b = len(ca(1)(1:2)//ca(2)(2:3)//ca(3)(3:4)) == 6
  logical, parameter :: test_07a = ca(1)(:2) == "ab"
  logical, parameter :: test_07b = ca(1)(3:) == "cd"
  logical, parameter :: test_07c = ca(1)(:-1) == ""
  logical, parameter :: test_07d = ca(1)(5:) == ""
  logical, parameter :: test_07e = ca(1)(:) == "abcd"
end module

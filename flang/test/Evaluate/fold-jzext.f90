! RUN: %python %S/test_folding.py %s %flang_fc1
! Tests folding of IZEXT() & JZEXT()
module m
  logical, parameter :: test_1 = kind(izext(-1_1)) == 2
  logical, parameter :: test_2 = izext(-1_1) == 255_2
  logical, parameter :: test_3 = kind(jzext(-1_1)) == 4
  logical, parameter :: test_4 = jzext(-1_1) == 255_4
  logical, parameter :: test_5 = kind(jzext(-1_2)) == 4
  logical, parameter :: test_6 = jzext(-1_2) == 255_4
end module

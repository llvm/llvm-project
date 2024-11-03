! RUN: %python %S/test_folding.py %s %flang_fc1
! Exercise parsing, expression analysis, and folding on a very tall expression tree
! 32*32 = 1024 repetitions
#define M0(x) x+x+x+x+x+x+x+x+x+x+x+x+x+x+x+x+x+x+x+x+x+x+x+x+x+x+x+x+x+x+x+x
#define M1(x) x+x+x+x+x+x+x+x+x+x+x+x+x+x+x+x+x+x+x+x+x+x+x+x+x+x+x+x+x+x+x+x
module m
  logical, parameter :: test_1 = 32**2 .EQ. M1(M0(1))
end module

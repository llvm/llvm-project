! RUN: %python %S/test_folding.py %s %flang_fc1
! Tests folding of BTEST
module m1
  integer, parameter :: ia1(*) = [(j, j=0, 15)]
  logical, parameter :: test_ia1a = all(btest(ia1, 0) .eqv. [(.false., .true., j=1, 8)])
  logical, parameter :: test_ia1b = all(btest(ia1, 1) .eqv. [(.false., .false., .true., .true., j=1, 4)])
  logical, parameter :: test_ia1c = all(btest(ia1, 2) .eqv. [(modulo(j/4, 2) == 1, j=0, 15)])
  logical, parameter :: test_ia1d = all(btest(ia1, 3) .eqv. [(j > 8, j=1, 16)])
  logical, parameter :: test_shft1 = all([(btest(ishft(1_1, j), j), j=0, 7)])
  logical, parameter :: test_shft2 = all([(btest(ishft(1_2, j), j), j=0, 15)])
  logical, parameter :: test_shft4 = all([(btest(ishft(1_4, j), j), j=0, 31)])
  logical, parameter :: test_shft8 = all([(btest(ishft(1_8, j), j), j=0, 63)])
  logical, parameter :: test_shft16 = all([(btest(ishft(1_16, j), j), j=0, 127)])
  logical, parameter :: test_set1 = all([(btest(ibset(0_1, j), j), j=0, 7)])
  logical, parameter :: test_set2 = all([(btest(ibset(0_2, j), j), j=0, 15)])
  logical, parameter :: test_set4 = all([(btest(ibset(0_4, j), j), j=0, 31)])
  logical, parameter :: test_set8 = all([(btest(ibset(0_8, j), j), j=0, 63)])
  logical, parameter :: test_set16 = all([(btest(ibset(0_16, j), j), j=0, 127)])
  logical, parameter :: test_z = .not. any([(btest(0_4, j), j=0, 31)])
  logical, parameter :: test_shft1e = all(btest([(ishft(1_1, j), j=0, 7)], [(j, j=0, 7)]))
end module

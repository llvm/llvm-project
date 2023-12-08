! RUN: %python %S/test_folding.py %s %flang_fc1

! Test combined shift intrinsics

module consts
  integer(1), parameter :: z40_1 = 64_1

end module consts

module dshiftltest
  use consts

  logical, parameter :: test_lzzz = dshiftl(0, 0, 0) .EQ. 0
  logical, parameter :: test_lzzn = dshiftl(0, 0, 12) .EQ. 0
  logical, parameter :: test_lnzn = dshiftl(36_1, 0_1, 5) .EQ. -128_2
  logical, parameter :: test_lznn = dshiftl(0, 33, 12) .EQ. 0

  logical, parameter :: test_l0 = dshiftl(3, 4, 0) .EQ. 3
  logical, parameter :: test_l32 = dshiftl(3, 4, 32) .EQ. 4
  logical, parameter :: test_l1 = dshiftl(9, 7, 1) .EQ. 18
  logical, parameter :: test_l2 = dshiftl(15_1, z40_1, 2) .EQ. 61
  logical, parameter :: test_le1 = dshiftl(64_1, 4_1, 7) .EQ. 2

  logical, parameter :: test_nb = dshiftl(15_1, Z'40', 2) .EQ. 61
  logical, parameter :: test_nbo = dshiftl(15_1, Z'F40', 2) .EQ. 61
end module dshiftltest

module dshiftrtest
  use consts

  logical, parameter :: test_rzzz = dshiftr(0, 0, 0) .EQ. 0
  logical, parameter :: test_rzzn = dshiftr(0, 0, 12) .EQ. 0
  logical, parameter :: test_rnzn = dshiftr(36_1, 0_1, 5) .EQ. 32_1
  logical, parameter :: test_rznn = dshiftr(0, 33, 12) .EQ. 0

  logical, parameter :: test_r0 = dshiftr(3, 4, 0) .EQ. 4
  logical, parameter :: test_r32 = dshiftr(3, 4, 32) .EQ. 3
  logical, parameter :: test_r1 = dshiftr(4, 17, 1) .EQ. 8
  logical, parameter :: test_r2 = dshiftr(15_1, z40_1, 2) .EQ. -48_1
  logical, parameter :: test_re1 = dshiftr(64_1, 4_1, 7) .EQ. -128_2

  logical, parameter :: test_nb = dshiftr(15_1, Z'40', 2) .EQ. -48_1
  logical, parameter :: test_nbo = dshiftr(15_1, Z'F40', 2) .EQ. -48_1
end module dshiftrtest

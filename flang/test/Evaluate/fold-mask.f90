! RUN: %python %S/test_folding.py %s %flang_fc1

! Test mask intrinsic folding

module maskltest
  logical, parameter :: test_l0 = maskl(0) .EQ. 0
  logical, parameter :: test_l1 = maskl(1) .EQ. -2147483648_8
  logical, parameter :: test_l2 = maskl(2) .EQ. -1073741824
  logical, parameter :: test_lm = maskl(16) .EQ. -65536
  logical, parameter :: test_le1 = maskl(31) .EQ. -2
  logical, parameter :: test_le = maskl(32) .EQ. -1
  logical, parameter :: test_lo = maskl(33) .EQ. -1

  logical, parameter :: test_l0_1 = maskl(0, 1) .EQ. 0_1
  logical, parameter :: test_l1_1 = maskl(1, 1) .EQ. -128_2
  logical, parameter :: test_l2_1 = maskl(2, 1) .EQ. -64_1
  logical, parameter :: test_lm_1 = maskl(4, 1) .EQ. -16_1
  logical, parameter :: test_le1_1 = maskl(7, 1) .EQ. -2_1
  logical, parameter :: test_le_1 = maskl(8, 1) .EQ. -1_1
  logical, parameter :: test_lo_1 = maskl(9, 1) .EQ. -1_1

  logical, parameter :: test_l0_2 = maskl(0, 2) .EQ. 0_2
  logical, parameter :: test_l1_2 = maskl(1, 2) .EQ. -32768_4
  logical, parameter :: test_l2_2 = maskl(2, 2) .EQ. -16384_2
  logical, parameter :: test_lm_2 = maskl(8, 2) .EQ. -256_2
  logical, parameter :: test_le1_2 = maskl(15, 2) .EQ. -2_2
  logical, parameter :: test_le_2 = maskl(16, 2) .EQ. -1_2
  logical, parameter :: test_lo_2 = maskl(17, 2) .EQ. -1_2

  logical, parameter :: test_l0_4 = maskl(0, 4) .EQ. 0_4
  logical, parameter :: test_l1_4 = maskl(1, 4) .EQ. -2147483648_8
  logical, parameter :: test_l2_4 = maskl(2, 4) .EQ. -1073741824_4
  logical, parameter :: test_lm_4 = maskl(16, 4) .EQ. -65536_4
  logical, parameter :: test_le1_4 = maskl(31, 4) .EQ. -2_4
  logical, parameter :: test_le_4 = maskl(32, 4) .EQ. -1_4
  logical, parameter :: test_lo_4 = maskl(33, 4) .EQ. -1_4

  logical, parameter :: test_l0_8 = maskl(0, 8) .EQ. 0_8
  logical, parameter :: test_l1_8 = maskl(1, 8) .EQ. -9223372036854775808_16
  logical, parameter :: test_l2_8 = maskl(2, 8) .EQ. -4611686018427387904_8
  logical, parameter :: test_lm_8 = maskl(32, 8) .EQ. -4294967296_8
  logical, parameter :: test_le1_8 = maskl(63, 8) .EQ. -2_8
  logical, parameter :: test_le_8 = maskl(64, 8) .EQ. -1_8
  logical, parameter :: test_lo_8 = maskl(65, 8) .EQ. -1_8

  logical, parameter :: test_l0_16 = maskl(0, 16) .EQ. 0_16
  logical, parameter :: test_l2_16 = maskl(2, 16) .EQ. -85070591730234615865843651857942052864_16
  logical, parameter :: test_lm_16 = maskl(64, 16) .EQ. -18446744073709551616_16
  logical, parameter :: test_le1_16 = maskl(127, 16) .EQ. -2_16
  logical, parameter :: test_le_16 = maskl(128, 16) .EQ. -1_16
  logical, parameter :: test_lo_16 = maskl(129, 16) .EQ. -1_16
end module maskltest

module maskrtest
  logical, parameter :: test_r0 = maskr(0) .EQ. 0
  logical, parameter :: test_r1 = maskr(1) .EQ. 1
  logical, parameter :: test_r2 = maskr(2) .EQ. 3
  logical, parameter :: test_rm = maskr(16) .EQ. 65535
  logical, parameter :: test_re1 = maskr(31) .EQ. 2147483647
  logical, parameter :: test_re = maskr(32) .EQ. -1
  logical, parameter :: test_ro = maskr(33) .EQ. -1

  logical, parameter :: test_r0_1 = maskr(0, 1) .EQ. 0_1
  logical, parameter :: test_r1_1 = maskr(1, 1) .EQ. 1_1
  logical, parameter :: test_r2_1 = maskr(2, 1) .EQ. 3_1
  logical, parameter :: test_rm_1 = maskr(4, 1) .EQ. 15_1
  logical, parameter :: test_re1_1 = maskr(7, 1) .EQ. 127_1
  logical, parameter :: test_re_1 = maskr(8, 1) .EQ. -1_1
  logical, parameter :: test_ro_1 = maskr(9, 1) .EQ. -1_1

  logical, parameter :: test_r0_2 = maskr(0, 2) .EQ. 0_2
  logical, parameter :: test_r1_2 = maskr(1, 2) .EQ. 1_2
  logical, parameter :: test_r2_2 = maskr(2, 2) .EQ. 3_2
  logical, parameter :: test_rm_2 = maskr(8, 2) .EQ. 255_2
  logical, parameter :: test_re1_2 = maskr(15, 2) .EQ. 32767_2
  logical, parameter :: test_re_2 = maskr(16, 2) .EQ. -1_2
  logical, parameter :: test_ro_2 = maskr(17, 2) .EQ. -1_2

  logical, parameter :: test_r0_4 = maskr(0, 4) .EQ. 0_4
  logical, parameter :: test_r1_4 = maskr(1, 4) .EQ. 1_4
  logical, parameter :: test_r2_4 = maskr(2, 4) .EQ. 3_4
  logical, parameter :: test_rm_4 = maskr(16, 4) .EQ. 65535_4
  logical, parameter :: test_re1_4 = maskr(31, 4) .EQ. 2147483647_4
  logical, parameter :: test_re_4 = maskr(32, 4) .EQ. -1_4
  logical, parameter :: test_ro_4 = maskr(33, 4) .EQ. -1_4

  logical, parameter :: test_r0_8 = maskr(0, 8) .EQ. 0_8
  logical, parameter :: test_r1_8 = maskr(1, 8) .EQ. 1_8
  logical, parameter :: test_r2_8 = maskr(2, 8) .EQ. 3_8
  logical, parameter :: test_rm_8 = maskr(32, 8) .EQ. 4294967295_8
  logical, parameter :: test_re1_8 = maskr(63, 8) .EQ. 9223372036854775807_8
  logical, parameter :: test_re_8 = maskr(64, 8) .EQ. -1_8
  logical, parameter :: test_ro_8 = maskr(65, 8) .EQ. -1_8

  logical, parameter :: test_r0_16 = maskr(0, 16) .EQ. 0_16
  logical, parameter :: test_r1_16 = maskr(1, 16) .EQ. 1_16
  logical, parameter :: test_r2_16 = maskr(2, 16) .EQ. 3_16
  logical, parameter :: test_rm_16 = maskr(64, 16) .EQ. 18446744073709551615_16
  logical, parameter :: test_re1_16 = maskr(127, 16) .EQ. 170141183460469231731687303715884105727_16
  logical, parameter :: test_re_16 = maskr(128, 16) .EQ. -1_16
  logical, parameter :: test_ro_16 = maskr(129, 16) .EQ. -1_16
end module maskrtest

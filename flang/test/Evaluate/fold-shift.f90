! RUN: %python %S/test_folding.py %s %flang_fc1

! Test directional shift intrinsics

module consts
  integer, parameter :: z40 = 1073741824
  integer, parameter :: z60 = 1610612736
  integer, parameter :: z80 = -2147483648_8
  integer, parameter :: zC0 = -1073741824
  integer, parameter :: zE0 = -536870912
  integer, parameter :: zF0 = -268435456

  integer(1), parameter :: z40_1 = 64_1
  integer(1), parameter :: z60_1 = 96_1
  integer(1), parameter :: z80_1 = -128_2
  integer(1), parameter :: zC0_1 = -64_1
  integer(1), parameter :: zE0_1 = -32_1
  integer(1), parameter :: zF0_1 = -16_1

  integer(2), parameter :: z40_2 = 16384_2
  integer(2), parameter :: z60_2 = 24576_2
  integer(2), parameter :: z80_2 = -32768_4
  integer(2), parameter :: zC0_2 = -16384_2
  integer(2), parameter :: zE0_2 = -8192_2
  integer(2), parameter :: zF0_2 = -4096_2

  integer(4), parameter :: z40_4 = 1073741824_4
  integer(4), parameter :: z60_4 = 1610612736_4
  integer(4), parameter :: z80_4 = -2147483648_8
  integer(4), parameter :: zC0_4 = -1073741824_4
  integer(4), parameter :: zE0_4 = -536870912_4
  integer(4), parameter :: zF0_4 = -268435456_4

  integer(8), parameter :: z40_8 = 4611686018427387904_8
  integer(8), parameter :: z60_8 = 6917529027641081856_8
  integer(8), parameter :: z80_8 = -9223372036854775808_16
  integer(8), parameter :: zC0_8 = -4611686018427387904_8
  integer(8), parameter :: zE0_8 = -2305843009213693952_8
  integer(8), parameter :: ZF0_8 = -1152921504606846976_8

  integer(16), parameter :: z40_16 = 85070591730234615865843651857942052864_16
  integer(16), parameter :: z60_16 = 127605887595351923798765477786913079296_16
  integer(16), parameter :: zC0_16 = -85070591730234615865843651857942052864_16
  integer(16), parameter :: zE0_16 = -42535295865117307932921825928971026432_16
  integer(16), parameter :: ZF0_16 = -21267647932558653966460912964485513216_16

end module consts

module shiftltest
  use consts

  logical, parameter :: test_l0 = shiftl(1, 0) .EQ. 1
  logical, parameter :: test_l1 = shiftl(1, 1) .EQ. 2
  logical, parameter :: test_lm = shiftl(1, 16) .EQ. 65536
  logical, parameter :: test_l3e2 = shiftl(3, 30) .EQ. zC0
  logical, parameter :: test_l3e1 = shiftl(3, 31) .EQ. z80
  logical, parameter :: test_le1 = shiftl(1, 31) .EQ. z80
  logical, parameter :: test_le = shiftl(1, 32) .EQ. 0

  logical, parameter :: test_l0_1 = shiftl(1_1, 0) .EQ. 1_1
  logical, parameter :: test_l1_1 = shiftl(1_1, 1) .EQ. 2_1
  logical, parameter :: test_lm_1 = shiftl(1_1, 4) .EQ. 16_1
  logical, parameter :: test_l3e2_1 = shiftl(3_1, 6) .EQ. zC0_1
  logical, parameter :: test_l3e1_1 = shiftl(3_1, 7) .EQ. z80_1
  logical, parameter :: test_le1_1 = shiftl(1_1, 7) .EQ. z80_1
  logical, parameter :: test_le_1 = shiftl(1_1, 8) .EQ. 0_1

  logical, parameter :: test_l0_2 = shiftl(1_2, 0) .EQ. 1
  logical, parameter :: test_l1_2 = shiftl(1_2, 1) .EQ. 2
  logical, parameter :: test_lm_2 = shiftl(1_2, 8) .EQ. 256_2
  logical, parameter :: test_l3e2_2 = shiftl(3_2, 14) .EQ. zC0_2
  logical, parameter :: test_l3e1_2 = shiftl(3_2, 15) .EQ. z80_2
  logical, parameter :: test_le1_2 = shiftl(1_2, 15) .EQ. z80_2
  logical, parameter :: test_le_2 = shiftl(1_2, 16) .EQ. 0_2

  logical, parameter :: test_l0_4 = shiftl(1_4, 0) .EQ. 1_4
  logical, parameter :: test_l1_4 = shiftl(1_4, 1) .EQ. 2_4
  logical, parameter :: test_lm_4 = shiftl(1_4, 16) .EQ. 65536_4
  logical, parameter :: test_l3e2_4 = shiftl(3_4, 30) .EQ. zC0_4
  logical, parameter :: test_l3e1_4 = shiftl(3_4, 31) .EQ. z80_4
  logical, parameter :: test_le1_4 = shiftl(1_4, 31) .EQ. z80_4
  logical, parameter :: test_le_4 = shiftl(1_4, 32) .EQ. 0_4

  logical, parameter :: test_l0_8 = shiftl(1_8, 0) .EQ. 1_8
  logical, parameter :: test_l1_8 = shiftl(1_8, 1) .EQ. 2_8
  logical, parameter :: test_lm_8 = shiftl(1_8, 16) .EQ. 65536
  logical, parameter :: test_l3e2_8 = shiftl(3_8, 62) .EQ. zC0_8
  logical, parameter :: test_l3e1_8 = shiftl(3_8, 63) .EQ. z80_8
  logical, parameter :: test_le1_8 = shiftl(1_8, 63) .EQ. z80_8
  logical, parameter :: test_le_8 = shiftl(1_8, 64) .EQ. 0_8

  logical, parameter :: test_l0_16 = shiftl(1_16, 0) .EQ. 1_16
  logical, parameter :: test_l1_16 = shiftl(1_16, 1) .EQ. 2_16
  logical, parameter :: test_lm_16 = shiftl(1_16, 64) .EQ. 18446744073709551616_16
  logical, parameter :: test_l3e2_16 = shiftl(3_16, 126) .EQ. zC0_16
  logical, parameter :: test_le_16 = shiftl(1_16, 128) .EQ. 0_16
end module shiftltest

module shiftrtest
  use consts

  logical, parameter :: test_r0 = shiftr(zC0, 0) .EQ. zC0
  logical, parameter :: test_r1 = shiftr(zC0, 1) .EQ. z60
  logical, parameter :: test_rm = shiftr(z40, 16) .EQ. 16384
  logical, parameter :: test_r3e2 = shiftr(zC0, 30) .EQ. 3
  logical, parameter :: test_r3e1 = shiftr(zC0, 31) .EQ. 1
  logical, parameter :: test_re = shiftr(z80, 32) .EQ. 0

  logical, parameter :: test_r0_1 = shiftr(zC0_1, 0) .EQ. zC0_1
  logical, parameter :: test_r1_1 = shiftr(zC0_1, 1) .EQ. z60_1
  logical, parameter :: test_rm_1 = shiftr(z40_1, 4) .EQ. 4_1
  logical, parameter :: test_r3e2_1 = shiftr(zC0_1, 6) .EQ. 3_1
  logical, parameter :: test_r3e1_1 = shiftr(zC0_1, 7) .EQ. 1_1
  logical, parameter :: test_re_1 = shiftr(z80_1, 8) .EQ. 0_1

  logical, parameter :: test_r0_2 = shiftr(zC0_2, 0) .EQ. zC0_2
  logical, parameter :: test_r1_2 = shiftr(zC0_2, 1) .EQ. z60_2
  logical, parameter :: test_rm_2 = shiftr(z40_2, 8) .EQ. 64_1
  logical, parameter :: test_r3e2_2 = shiftr(zC0_2, 14) .EQ. 3_2
  logical, parameter :: test_r3e1_2 = shiftr(zC0_2, 15) .EQ. 1_2
  logical, parameter :: test_re_2 = shiftr(z80_2, 16) .EQ. 0_2

  logical, parameter :: test_r0_4 = shiftr(zC0_4, 0) .EQ. zC0_4
  logical, parameter :: test_r1_4 = shiftr(zC0_4, 1) .EQ. z60_4
  logical, parameter :: test_rm_4 = shiftr(z40_4, 16) .EQ. 16384_4
  logical, parameter :: test_r3e2_4 = shiftr(zC0_4, 30) .EQ. 3_4
  logical, parameter :: test_r3e1_4 = shiftr(zC0_4, 31) .EQ. 1_4
  logical, parameter :: test_re_4 = shiftr(z80_4, 32) .EQ. 0_4

  logical, parameter :: test_r0_8 = shiftr(zC0_8, 0) .EQ. zC0_8
  logical, parameter :: test_r1_8 = shiftr(zC0_8, 1) .EQ. z60_8
  logical, parameter :: test_rm_8 = shiftr(z40_8, 32) .EQ. 1073741824_8
  logical, parameter :: test_r3e2_8 = shiftr(zC0_8, 62) .EQ. 3_8
  logical, parameter :: test_r3e1_8 = shiftr(zC0_8, 63) .EQ. 1_8
  logical, parameter :: test_re_8 = shiftr(z80_8, 64) .EQ. 0_8

  logical, parameter :: test_r0_16 = shiftr(zC0_16, 0) .EQ. zC0_16
  logical, parameter :: test_r1_16 = shiftr(zC0_16, 1) .EQ. z60_16
  logical, parameter :: test_rm_16 = shiftr(z40_16, 64) .EQ. 4611686018427387904_16
  logical, parameter :: test_r3e2_16 = shiftr(zC0_16, 126) .EQ. 3_16
  logical, parameter :: test_r3e1_16 = shiftr(zC0_16, 127) .EQ. 1_16
  logical, parameter :: test_re_16 = shiftr(z40_16, 128) .EQ. 0_16
end module shiftrtest

module shiftatest
  use consts

  logical, parameter :: test_a0 = shifta(zC0, 0) .EQ. zC0
  logical, parameter :: test_a1 = shifta(zC0, 1) .EQ. zE0
  logical, parameter :: test_a2 = shifta(zC0, 2) .EQ. zF0
  logical, parameter :: test_a3e2 = shifta(zC0, 29) .EQ. -2
  logical, parameter :: test_a3e1 = shifta(zC0, 31) .EQ. -1
  logical, parameter :: test_ae = shifta(z80, 32) .EQ. -1

  logical, parameter :: test_a0_1 = shifta(zC0_1, 0) .EQ. zC0_1
  logical, parameter :: test_a1_1 = shifta(zC0_1, 1) .EQ. zE0_1
  logical, parameter :: test_a2_1 = shifta(zC0_1, 2) .EQ. zF0_1
  logical, parameter :: test_a3e2_1 = shifta(zC0_1, 5) .EQ. -2_1
  logical, parameter :: test_a3e1_1 = shifta(zC0_1, 7) .EQ. -1_1
  logical, parameter :: test_ae_1 = shifta(z80_1, 8) .EQ. -1_1

  logical, parameter :: test_a0_2 = shifta(zC0_2, 0) .EQ. zC0_2
  logical, parameter :: test_a1_2 = shifta(zC0_2, 1) .EQ. zE0_2
  logical, parameter :: test_a2_2 = shifta(zC0_2, 2) .EQ. zF0_2
  logical, parameter :: test_a3e2_2 = shifta(zC0_2, 13) .EQ. -2_2
  logical, parameter :: test_a3e1_2 = shifta(zC0_2, 15) .EQ. -1_2
  logical, parameter :: test_ae_2 = shifta(z80_2, 16) .EQ. -1_2

  logical, parameter :: test_a0_4 = shifta(zC0_4, 0) .EQ. zC0_4
  logical, parameter :: test_a1_4 = shifta(zC0_4, 1) .EQ. zE0_4
  logical, parameter :: test_a2_4 = shifta(zC0_4, 2) .EQ. zF0_4
  logical, parameter :: test_a3e2_4 = shifta(zC0_4, 29) .EQ. -2_4
  logical, parameter :: test_a3e1_4 = shifta(zC0_4, 31) .EQ. -1_4
  logical, parameter :: test_ae_4 = shifta(z80_4, 32) .EQ. -1_4

  logical, parameter :: test_a0_8 = shifta(zC0_8, 0) .EQ. zC0_8
  logical, parameter :: test_a1_8 = shifta(zC0_8, 1) .EQ. zE0_8
  logical, parameter :: test_a2_8 = shifta(zC0_8, 2) .EQ. zF0_8
  logical, parameter :: test_a3e2_8 = shifta(zC0_8, 61) .EQ. -2_8
  logical, parameter :: test_a3e1_8 = shifta(zC0_8, 63) .EQ. -1_8
  logical, parameter :: test_ae_8 = shifta(z80_8, 64) .EQ. -1_8

  logical, parameter :: test_a0_16 = shifta(zC0_16, 0) .EQ. zC0_16
  logical, parameter :: test_a1_16 = shifta(zC0_16, 1) .EQ. zE0_16
  logical, parameter :: test_a2_16 = shifta(zC0_16, 2) .EQ. zF0_16
  logical, parameter :: test_a3e2_16 = shifta(zC0_16, 125) .EQ. -2_16
  logical, parameter :: test_a3e1_16 = shifta(zC0_16, 127) .EQ. -1_16
  logical, parameter :: test_ae_16 = shifta(zC0_16, 128) .EQ. -1_16
end module shiftatest

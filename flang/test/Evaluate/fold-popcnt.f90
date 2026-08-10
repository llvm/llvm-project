! RUN: %python %S/test_folding.py %s %flang_fc1
! Test counts of set bits.

module leadz_tests
  logical, parameter :: test_z1 = leadz(0_1) .EQ. 8
  logical, parameter :: test_o1 = leadz(1_1) .EQ. 7
  logical, parameter :: test_t1 = leadz(2_1) .EQ. 6
  logical, parameter :: test_f1 = leadz(15_1) .EQ. 4
  logical, parameter :: test_b1 = leadz(16_1) .EQ. 3
  logical, parameter :: test_m11 = leadz(-1_1) .EQ. 0
  logical, parameter :: test_m12 = leadz(-2_1) .EQ. 0
  logical, parameter :: test_mb1 = leadz(-120_1) .EQ. 0

  logical, parameter :: test_z2 = leadz(0_2) .EQ. 16
  logical, parameter :: test_o2 = leadz(1_2) .EQ. 15
  logical, parameter :: test_t2 = leadz(2_2) .EQ. 14
  logical, parameter :: test_f2 = leadz(15_2) .EQ. 12
  logical, parameter :: test_m21 = leadz(-1_2) .EQ. 0
  logical, parameter :: test_m22 = leadz(-2_2) .EQ. 0
  logical, parameter :: test_mb2 = leadz(-32640_2) .EQ. 0

  logical, parameter :: test_z4 = leadz(0_4) .EQ. 32
  logical, parameter :: test_o4 = leadz(1_4) .EQ. 31
  logical, parameter :: test_t4 = leadz(2_4) .EQ. 30
  logical, parameter :: test_f4 = leadz(15_4) .EQ. 28
  logical, parameter :: test_m41 = leadz(-1_4) .EQ. 0
  logical, parameter :: test_m42 = leadz(-2_4) .EQ. 0
  logical, parameter :: test_mb4 = leadz(-2147450880_4) .EQ. 0

  logical, parameter :: test_z8 = leadz(0_8) .EQ. 64
  logical, parameter :: test_o8 = leadz(1_8) .EQ. 63
  logical, parameter :: test_t8 = leadz(2_8) .EQ. 62
  logical, parameter :: test_f8 = leadz(15_8) .EQ. 60
  logical, parameter :: test_m81 = leadz(-1_8) .EQ. 0
  logical, parameter :: test_m82 = leadz(-2_8) .EQ. 0
  logical, parameter :: test_mb8 = leadz(-9223372034707292160_8) .EQ. 0

  logical, parameter :: test_z16 = leadz(0_16) .EQ. 128
  logical, parameter :: test_o16 = leadz(1_16) .EQ. 127
  logical, parameter :: test_t16 = leadz(2_16) .EQ. 126
  logical, parameter :: test_f16 = leadz(15_16) .EQ. 124
  logical, parameter :: test_m161 = leadz(-1_16) .EQ. 0
  logical, parameter :: test_m162 = leadz(-2_16) .EQ. 0
  logical, parameter :: test_mb16 = leadz(18446744073709551616_16) .EQ. 63
end module leadz_tests

module trailz_tests
  logical, parameter :: test_z1 = trailz(0_1) .EQ. 8
  logical, parameter :: test_o1 = trailz(1_1) .EQ. 0
  logical, parameter :: test_t1 = trailz(2_1) .EQ. 1
  logical, parameter :: test_f1 = trailz(15_1) .EQ. 0
  logical, parameter :: test_b1 = trailz(16_1) .EQ. 4
  logical, parameter :: test_m11 = trailz(-1_1) .EQ. 0
  logical, parameter :: test_m12 = trailz(-2_1) .EQ. 1
  logical, parameter :: test_mb1 = trailz(-120_1) .EQ. 3

  logical, parameter :: test_z2 = trailz(0_2) .EQ. 16
  logical, parameter :: test_o2 = trailz(1_2) .EQ. 0
  logical, parameter :: test_t2 = trailz(2_2) .EQ. 1
  logical, parameter :: test_f2 = trailz(15_2) .EQ. 0
  logical, parameter :: test_m21 = trailz(-1_2) .EQ. 0
  logical, parameter :: test_m22 = trailz(-2_2) .EQ. 1
  logical, parameter :: test_mb2 = trailz(-32640_2) .EQ. 7

  logical, parameter :: test_z4 = trailz(0_4) .EQ. 32
  logical, parameter :: test_o4 = trailz(1_4) .EQ. 0
  logical, parameter :: test_t4 = trailz(2_4) .EQ. 1
  logical, parameter :: test_f4 = trailz(15_4) .EQ. 0
  logical, parameter :: test_m41 = trailz(-1_4) .EQ. 0
  logical, parameter :: test_m42 = trailz(-2_4) .EQ. 1
  logical, parameter :: test_mb4 = trailz(-2147450880_4) .EQ. 15

  logical, parameter :: test_z8 = trailz(0_8) .EQ. 64
  logical, parameter :: test_o8 = trailz(1_8) .EQ. 0
  logical, parameter :: test_t8 = trailz(2_8) .EQ. 1
  logical, parameter :: test_f8 = trailz(15_8) .EQ. 0
  logical, parameter :: test_m81 = trailz(-1_8) .EQ. 0
  logical, parameter :: test_m82 = trailz(-2_8) .EQ. 1
  logical, parameter :: test_mb8 = trailz(-9223372034707292160_8) .EQ. 31

  logical, parameter :: test_z16 = trailz(0_16) .EQ. 128
  logical, parameter :: test_o16 = trailz(1_16) .EQ. 0
  logical, parameter :: test_t16 = trailz(2_16) .EQ. 1
  logical, parameter :: test_f16 = trailz(15_16) .EQ. 0
  logical, parameter :: test_m161 = trailz(-1_16) .EQ. 0
  logical, parameter :: test_m162 = trailz(-2_16) .EQ. 1
  logical, parameter :: test_mb16 = trailz(18446744073709551616_16) .EQ. 64
end module trailz_tests

module popcnt_tests
  logical, parameter :: test_z1 = popcnt(0_1) .EQ. 0
  logical, parameter :: test_o1 = popcnt(1_1) .EQ. 1
  logical, parameter :: test_t1 = popcnt(2_1) .EQ. 1
  logical, parameter :: test_f1 = popcnt(15_1) .EQ. 4
  logical, parameter :: test_b1 = popcnt(16_1) .EQ. 1
  logical, parameter :: test_m11 = popcnt(-1_1) .EQ. 8
  logical, parameter :: test_m12 = popcnt(-2_1) .EQ. 7
  logical, parameter :: test_mb1 = popcnt(-120_1) .EQ. 2

  logical, parameter :: test_z2 = popcnt(0_2) .EQ. 0
  logical, parameter :: test_o2 = popcnt(1_2) .EQ. 1
  logical, parameter :: test_t2 = popcnt(2_2) .EQ. 1
  logical, parameter :: test_f2 = popcnt(15_2) .EQ. 4
  logical, parameter :: test_m21 = popcnt(-1_2) .EQ. 16
  logical, parameter :: test_m22 = popcnt(-2_2) .EQ. 15
  logical, parameter :: test_mb2 = popcnt(-32640_2) .EQ. 2

  logical, parameter :: test_z4 = popcnt(0_4) .EQ. 0
  logical, parameter :: test_o4 = popcnt(1_4) .EQ. 1
  logical, parameter :: test_t4 = popcnt(2_4) .EQ. 1
  logical, parameter :: test_f4 = popcnt(15_4) .EQ. 4
  logical, parameter :: test_m41 = popcnt(-1_4) .EQ. 32
  logical, parameter :: test_m42 = popcnt(-2_4) .EQ. 31
  logical, parameter :: test_mb4 = popcnt(-2147450880_4) .EQ. 2

  logical, parameter :: test_z8 = popcnt(0_8) .EQ. 0
  logical, parameter :: test_o8 = popcnt(1_8) .EQ. 1
  logical, parameter :: test_t8 = popcnt(2_8) .EQ. 1
  logical, parameter :: test_f8 = popcnt(15_8) .EQ. 4
  logical, parameter :: test_m81 = popcnt(-1_8) .EQ. 64
  logical, parameter :: test_m82 = popcnt(-2_8) .EQ. 63
  logical, parameter :: test_mb8 = popcnt(-9223372034707292160_8) .EQ. 2

  logical, parameter :: test_z16 = popcnt(0_16) .EQ. 0
  logical, parameter :: test_o16 = popcnt(1_16) .EQ. 1
  logical, parameter :: test_t16 = popcnt(2_16) .EQ. 1
  logical, parameter :: test_f16 = popcnt(15_16) .EQ. 4
  logical, parameter :: test_m161 = popcnt(-1_16) .EQ. 128
  logical, parameter :: test_m162 = popcnt(-2_16) .EQ. 127
  logical, parameter :: test_mb16 = popcnt(18446744073709551616_16) .EQ. 1
end module popcnt_tests

module poppar_tests
  logical, parameter :: test_z1 = poppar(0_1) .EQ. 0
  logical, parameter :: test_o1 = poppar(1_1) .EQ. 1
  logical, parameter :: test_t1 = poppar(2_1) .EQ. 1
  logical, parameter :: test_f1 = poppar(15_1) .EQ. 0
  logical, parameter :: test_b1 = poppar(16_1) .EQ. 1
  logical, parameter :: test_m11 = poppar(-1_1) .EQ. 0
  logical, parameter :: test_m12 = poppar(-2_1) .EQ. 1
  logical, parameter :: test_mb1 = poppar(-120_1) .EQ. 0

  logical, parameter :: test_z2 = poppar(0_2) .EQ. 0
  logical, parameter :: test_o2 = poppar(1_2) .EQ. 1
  logical, parameter :: test_t2 = poppar(2_2) .EQ. 1
  logical, parameter :: test_f2 = poppar(15_2) .EQ. 0
  logical, parameter :: test_m21 = poppar(-1_2) .EQ. 0
  logical, parameter :: test_m22 = poppar(-2_2) .EQ. 1
  logical, parameter :: test_mb2 = poppar(-32640_2) .EQ. 0

  logical, parameter :: test_z4 = poppar(0_4) .EQ. 0
  logical, parameter :: test_o4 = poppar(1_4) .EQ. 1
  logical, parameter :: test_t4 = poppar(2_4) .EQ. 1
  logical, parameter :: test_f4 = poppar(15_4) .EQ. 0
  logical, parameter :: test_m41 = poppar(-1_4) .EQ. 0
  logical, parameter :: test_m42 = poppar(-2_4) .EQ. 1
  logical, parameter :: test_mb4 = poppar(-2147450880_4) .EQ. 0

  logical, parameter :: test_z8 = poppar(0_8) .EQ. 0
  logical, parameter :: test_o8 = poppar(1_8) .EQ. 1
  logical, parameter :: test_t8 = poppar(2_8) .EQ. 1
  logical, parameter :: test_f8 = poppar(15_8) .EQ. 0
  logical, parameter :: test_m81 = poppar(-1_8) .EQ. 0
  logical, parameter :: test_m82 = poppar(-2_8) .EQ. 1
  logical, parameter :: test_mb8 = poppar(-9223372034707292160_8) .EQ. 0

  logical, parameter :: test_z16 = poppar(0_16) .EQ. 0
  logical, parameter :: test_o16 = poppar(1_16) .EQ. 1
  logical, parameter :: test_t16 = poppar(2_16) .EQ. 1
  logical, parameter :: test_f16 = poppar(15_16) .EQ. 0
  logical, parameter :: test_m161 = poppar(-1_16) .EQ. 0
  logical, parameter :: test_m162 = poppar(-2_16) .EQ. 1
  logical, parameter :: test_mb16 = poppar(18446744073709551616_16) .EQ. 1
end module poppar_tests

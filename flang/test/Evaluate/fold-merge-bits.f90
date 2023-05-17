! RUN: %python %S/test_folding.py %s %flang_fc1

! Test fold merge_bits intrinsic

module merge_bitstest
  logical, parameter :: test_1 = merge_bits(13_1, 18_1, 22_1) .EQ. 4_1
  logical, parameter :: test_2 = merge_bits(13_2, 18_2, 22_2) .EQ. 4_2
  logical, parameter :: test_4 = merge_bits(13_4, 18_4, 22_4) .EQ. 4_4
  logical, parameter :: test_8 = merge_bits(13_8, 18_8, 22_8) .EQ. 4_8
  logical, parameter :: test_16 = merge_bits(13_16, 18_16, 22_16) .EQ. 4_16

  logical, parameter :: test_z11 = merge_bits(13_1, B'00010010', 22_1) .EQ. 4_1
  logical, parameter :: test_z12 = merge_bits(13_2, B'00010010', 22_2) .EQ. 4_2
  logical, parameter :: test_z14 = merge_bits(13_4, B'00010010', 22_4) .EQ. 4_4
  logical, parameter :: test_z18 = merge_bits(13_8, B'00010010', 22_8) .EQ. 4_8
  logical, parameter :: test_z116 = merge_bits(13_16, B'00010010', 22_16) .EQ. 4_16

  logical, parameter :: test_z01 = merge_bits(Z'0D', 18_1, 22_1) .EQ. 4_1
  logical, parameter :: test_z02 = merge_bits(Z'0D', 18_2, 22_2) .EQ. 4_2
  logical, parameter :: test_z04 = merge_bits(Z'0D', 18_4, 22_4) .EQ. 4_4
  logical, parameter :: test_z08 = merge_bits(Z'0D', 18_8, 22_8) .EQ. 4_8
  logical, parameter :: test_z016 = merge_bits(Z'0D', 18_16, 22_16) .EQ. 4_16
end module merge_bitstest

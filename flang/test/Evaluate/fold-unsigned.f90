! RUN: %python %S/test_folding.py %s %flang_fc1 -funsigned
! UNSIGNED operations and intrinsic functions

module m

  logical, parameter :: test_neg0    = -0u_1 == 0u_1
  logical, parameter :: test_neg0_k  = kind(-0u_1) == 1
  logical, parameter :: test_neg1    = -1u_1 == 255u
  logical, parameter :: test_neg255  = -255u_1 == 1u
  logical, parameter :: test_add1    = 0u_1 + 1u_1 == 1u_1
  logical, parameter :: test_add1_k  = kind(0u_1 + 1u_1) == 1
  logical, parameter :: test_addprom = 255u_1 + 1u == 256u
  logical, parameter :: test_addmix  = 255u_1 + z'1' == 0u
  logical, parameter :: test_sub1    = 0u_1 - 1u_1 == 255u_1
  logical, parameter :: test_sub1_k  = kind(0u_1 + 1u_1) == 1
  logical, parameter :: test_mul15   = 15u_1 * 15u_1 == 225u_1
  logical, parameter :: test_mul15_k = kind(15u_1 * 15u_1) == 1
  logical, parameter :: test_mul152  = 5u_1 * 52u_1 == 4u_1
  logical, parameter :: test_div15   = 225u_1 / 15u_1 == 15u_1
  logical, parameter :: test_div15_k = kind(225u_1 / 15u_1) == 1

  logical, parameter :: test_rel = all([0u_1 < 255u_1, 255u_1 > 0u_1, &
                                        0u_1 <= 255u_1, 255u_1 >= 0u_1])

  logical, parameter :: test_cus0    = int(0u,1) == 0
  logical, parameter :: test_cus0_k  = kind(int(0u,1)) == 1
  !WARN: warning: conversion of 255_U1 to INTEGER(1) overflowed; result is -1
  logical, parameter :: test_cus255  = int(255u_1,1) == -1
  logical, parameter :: test_cur255  = real(255u) == 255.

  logical, parameter :: test_csu255   = uint(255,1) == 255u_1
  logical, parameter :: test_csu255_k = kind(uint(255,1)) == 1
  logical, parameter :: test_cru255   = uint(255.) == 255u
  logical, parameter :: test_ctu255   = uint(z'ff',1) == 255u_1
  logical, parameter :: test_ctu255_k = kind(uint(z'ff',1)) == 1

  logical, parameter :: test_not1a = not(0u_1) == 255u_1
  logical, parameter :: test_not1b = not(255u_1) == 0u_1
  logical, parameter :: test_not4a = not(0u) == huge(0u)
  logical, parameter :: test_not4b = not(huge(0u)) == 0u

  logical, parameter :: test_iand1  = iand(170u,240u) == 160u
  logical, parameter :: test_ior1   = ior(170u,240u) == 250u
  logical, parameter :: test_ieor1  = ieor(170u,240u) == 90u
  logical, parameter :: test_ibclr1 = all(ibclr(255u,[(j,j=7,0,-1)]) == &
                                        [127u,191u,223u,239u, &
                                         247u,251u,253u,254u])
  logical, parameter :: test_ibset1 = all(ibset(0u,[(j,j=7,0,-1)]) == &
                                        [128u,64u,32u,16u,8u,4u,2u,1u])
  logical, parameter :: test_ibits1 = all(ibits(126u,[(j,j=0,7)],3) == &
                                        [6u,7u,7u,7u,7u,3u,1u,0u])

  logical, parameter :: test_mb_1 = merge_bits(13u_1, 18u_1, 22u_1) .EQ. 4u_1
  logical, parameter :: test_mb_2 = merge_bits(13u_2, 18u_2, 22u_2) .EQ. 4u_2
  logical, parameter :: test_mb_4 = merge_bits(13u_4, 18u_4, 22u_4) .EQ. 4u_4
  logical, parameter :: test_mb_8 = merge_bits(13u_8, 18u_8, 22u_8) .EQ. 4u_8
  logical, parameter :: test_mb_16 = merge_bits(13u_16, 18u_16, 22u_16) .EQ. 4u_16

  logical, parameter :: test_mb_z11 = merge_bits(13u_1, B'00010010', 22u_1) .EQ. 4u_1
  logical, parameter :: test_mb_z12 = merge_bits(13u_2, B'00010010', 22u_2) .EQ. 4u_2
  logical, parameter :: test_mb_z14 = merge_bits(13u_4, B'00010010', 22u_4) .EQ. 4u_4
  logical, parameter :: test_mb_z18 = merge_bits(13u_8, B'00010010', 22u_8) .EQ. 4u_8
  logical, parameter :: test_mb_z116 = merge_bits(13u_16, B'00010010', 22u_16) .EQ. 4u_16

  logical, parameter :: test_mb_z01 = merge_bits(Z'0D', 18u_1, 22u_1) .EQ. 4u_1
  logical, parameter :: test_mb_z02 = merge_bits(Z'0D', 18u_2, 22u_2) .EQ. 4u_2
  logical, parameter :: test_mb_z04 = merge_bits(Z'0D', 18u_4, 22u_4) .EQ. 4u_4
  logical, parameter :: test_mb_z08 = merge_bits(Z'0D', 18u_8, 22u_8) .EQ. 4u_8
  logical, parameter :: test_mb_z016 = merge_bits(Z'0D', 18u_16, 22u_16) .EQ. 4u_16

  logical, parameter :: test_btest1 = all(btest(uint(b'00011011'),[(j,j=0,7)]) .eqv. &
                                        [.true., .true., .false., .true., &
                                         .true., .false., .false., .false.])

  logical, parameter :: test_ishft1 = all(ishft(1u_1,[(j,j=0,8)]) == &
                                        [1u, 2u, 4u, 8u, 16u, 32u, 64u, 128u, 0u])
  logical, parameter :: test_ishft2 = all(ishft(255u,[(j,j=0,-8,-1)]) == &
                                        [255u, 127u, 63u, 31u, 15u, 7u, 3u, 1u, 0u])

  logical, parameter :: test_ishftc1 = all(ishftc(254u_1,[(j,j=0,8)]) == &
                                         [254u, 253u, 251u, 247u, 239u, 223u, 191u, 127u, 254u])
  logical, parameter :: test_ishftc2 = all(ishftc(254u_1,[(j,j=0,-8,-1)]) == &
                                         [254u, 127u, 191u, 223u, 239u, 247u, 251u, 253u, 254u])

  logical, parameter :: test_shifta1 = all(shifta(128u_1,[(j,j=0,8)]) == &
                                         [128u, 192u, 224u, 240u, 248u, 252u, 254u, 255u, 255u])
  logical, parameter :: test_shiftl1 = all(shiftl(1u_1,[(j,j=0,8)]) == &
                                        [1u, 2u, 4u, 8u, 16u, 32u, 64u, 128u, 0u])
  logical, parameter :: test_shiftr1 = all(shiftr(128u_1,[(j,j=0,8)]) == &
                                        [128u,64u,32u,16u,8u,4u,2u,1u,0u])
  logical, parameter :: test_shiftr2 = all(shiftr(255u,[(j,j=0,8)]) == &
                                        [255u, 127u, 63u, 31u, 15u, 7u, 3u, 1u, 0u])

  logical, parameter :: test_transfer1 = transfer(1.,0u) == uint(z'3f800000')
  logical, parameter :: test_transfer2 = transfer(uint(z'3f800000'),0.) == 1.

  logical, parameter :: test_bit_size = &
    all([integer::bit_size(0u_1), bit_size(0u_2), &
                  bit_size(0u_4), bit_size(0u_8), &
                  bit_size(0u_16)] == [8,16,32,64,128])

  logical, parameter :: test_digits = &
    all([digits(0u_1), digits(0u_2), digits(0u_4), digits(0u_8), &
         digits(0u_16)] == [8,16,32,64,128])

  logical, parameter :: test_huge_1  = huge(0u_1)  == 255u_1
  logical, parameter :: test_huge_2  = huge(0u_2)  == 65535u_2
  logical, parameter :: test_huge_4  = huge(0u_4)  == uint(huge(0_4),4) * 2u + 1u
  logical, parameter :: test_huge_8  = huge(0u_8)  == uint(huge(0_8),8) * 2u + 1u
  logical, parameter :: test_huge_16 = huge(0u_16) == uint(huge(0_16),16) * 2u + 1u

  logical, parameter :: test_range = &
    all([range(0u_1), range(0u_2), range(0u_4), range(0u_8), range(0u_16)] == &
        [2,4,9,19,38])

  logical, parameter :: test_max1  = max(0u,255u,128u) == 255u
  logical, parameter :: test_max1k = kind(max(0u_1,255u_1,128u_1)) == 1
  logical, parameter :: test_min1  = min(0u,255u,128u) == 0u
  logical, parameter :: test_min1k = kind(min(0u_1,255u_1,128u_1)) == 1
end

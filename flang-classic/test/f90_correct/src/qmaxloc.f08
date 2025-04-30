! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test MAXLOC intrinsic with quad-precision arguments

program p
  integer, parameter :: n = 9
  integer, parameter, dimension(1) :: init1 = maxloc((/4.5_16, 6.8_16, 6.8_16, 3.1_16/))
  integer, parameter, dimension(1) :: init2 = maxloc((/-4.5_16, -6.8_16, -6.8_16, -3.1_16/))
  integer, parameter, dimension(1) :: i_unop_huge = maxloc((/-huge(1.0_16), -huge(1.0_16), -huge(1.0_16), -huge(1.0_16)/))
  integer, parameter, dimension(1) :: i_unop_huge_b = maxloc((/-huge(1.0_16), -huge(1.0_16), -huge(1.0_16), -huge(1.0_16)/), back=.true.)
  real(16), dimension(2, 3) :: array_r16
  integer, dimension(2) :: tmp0
  integer, dimension(1) :: tmp1, tmp2, v_unop_huge, v_unop_huge_b
  integer :: rslts(n), expect(n)

  rslts = 0
  expect = 1

  !initialization
  if (init1(1) == 2) rslts(1) = 1
  if (init2(1) == 4) rslts(2) = 1
  if (i_unop_huge(1) == 0 .and. i_unop_huge_b(1) ==4) rslts(3) = 1

  !boundary
  v_unop_huge = maxloc((/-huge(1.0_16), -huge(1.0_16), -huge(1.0_16), -huge(1.0_16)/))
  v_unop_huge_b = maxloc((/-huge(1.0_16), -huge(1.0_16), -huge(1.0_16), -huge(1.0_16)/), back=.true.)
  if (v_unop_huge(1) == 1 .and. v_unop_huge_b(1) ==4) rslts(4) = 1
  
  tmp1 = maxloc((/tiny(1.0_16), tiny(1.0_16), tiny(1.0_16), tiny(1.0_16)/))
  tmp2 = maxloc((/tiny(1.0_16), tiny(1.0_16), tiny(1.0_16), tiny(1.0_16)/), back=.true.)
  if (tmp1(1) == 1 .and. tmp2(1) == 4) rslts(5) = 1

  tmp1 = maxloc((/epsilon(1.0_16), epsilon(1.0_16), epsilon(1.0_16), epsilon(1.0_16)/))
  tmp2 = maxloc((/epsilon(1.0_16), epsilon(1.0_16), epsilon(1.0_16), epsilon(1.0_16)/), back=.true.)
  if (tmp1(1) == 1 .and. tmp2(1) == 4) rslts(6) = 1

  !functional verification 
  tmp1 = maxloc((/1.000000000000000000000000000000_16, &
                  1.000000000000000000000000000002_16, &
                  1.000000000000000000000000000000_16, &
                  0.999999999999999999999999999999_16, &
                  1.000000000000000000000000000001_16, &
                  0.999999999999999999999999999998_16/))
  if (tmp1(1) == 2) rslts(7) = 1
  
  !variable param
  array_r16 = reshape((/1.222222222222222222222222222222_16, &
                        1.222222222222222222222222222221_16, &
                        1.222222222222222222222222222222_16, &
                        1.222222222222222222222222222220_16, &
                        1.222222222222222222222222222224_16, &
                        1.222222222222222222222222222223_16/), shape(array_r16))
  tmp0 = maxloc(array_r16)
  if (tmp0(1) == 1 .and. tmp0(2) == 3) rslts(8) = 1

  !support input 5 args
  tmp1 = maxloc((/-1.111111111111111111111111111111_16, &
                  -1.111111111111111111111111111112_16, &
                  -1.111111111111111111111111111110_16/), &
                dim = 1, mask = .true., kind = 1, back = .false. )
  if(tmp1(1) == 3) rslts(9) = 1

  call check(rslts, expect, n)
end


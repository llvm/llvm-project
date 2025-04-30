! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test CONJG intrinsic with quad-precision arguments

program test
  integer :: r(21), e(21)
  complex(16) :: zres(6) = [(1.02365154623254000000000000000000003_16, &
                             -1.36548946625488999999999999999999994_16), &
                            (huge(1.0_16), -huge(1.0_16)), &
                            (0.0_16, -0.0_16), &
                            (-huge(1.0_16), huge(1.0_16)), &
                            (tiny(1.0_16), -tiny(1.0_16)), &
                            (epsilon(1.0_16), -epsilon(1.0_16))]
  complex(16) :: z1(6) = [(1.02365154623254_16, &
                          1.36548946625489_16), &
                         (huge(1.0_16), huge(1.0_16)), &
                         (0.0_16, 0.0_16), &
                         (-huge(1.0_16), -huge(1.0_16)), &
                         (tiny(1.0_16), tiny(1.0_16)), &
                         (epsilon(1.0_16), epsilon(1.0_16))]
  complex(16),parameter :: z2(6) = [(1.02365154623254_16, &
                                    1.36548946625489_16), &
                                   (huge(1.0_16), huge(1.0_16)), &
                                   (0.0_16, 0.0_16), &
                                   (-huge(1.0_16), -huge(1.0_16)), &
                                   (tiny(1.0_16), tiny(1.0_16)), &
                                   (epsilon(1.0_16), epsilon(1.0_16))]
  complex(16) :: za1 = (1.02365154623254_16, 1.36548946625489_16), &
                 za2 = (huge(1.0_16), huge(1.0_16)), &
                 za3 = (0.0_16, 0.0_16), &
                 za4 = (-huge(1.0_16), -huge(1.0_16)), &
                 za5 = (tiny(1.0_16), tiny(1.0_16)), &
                 za6 = (epsilon(1.0_16), epsilon(1.0_16))
  complex(16), parameter :: zb1 = (1.02365154623254_16, 1.36548946625489_16), &
                            zb2 = (huge(1.0_16), huge(1.0_16)), &
                            zb3 = (0.0_16, 0.0_16), &
                            zb4 = (-huge(1.0_16), -huge(1.0_16)), &
                            zb5 = (tiny(1.0_16), tiny(1.0_16)), &
                            zb6 = (epsilon(1.0_16), epsilon(1.0_16))
  r = 0
  e = 1

  if (all(conjg(z1) .EQ. zres)) r(1) = 1

  if (all(conjg(z2) .EQ. zres)) r(2) = 1

  if (conjg((1.02365154623254_16, 1.36548946625489_16)) .EQ. zres(1)) r(3) = 1
  if (conjg((huge(1.0_16), huge(1.0_16))) .EQ. zres(2)) r(4) = 1
  if (conjg((0.0_16, 0.0_16)) .EQ. zres(3)) r(5) = 1
  if (conjg((-huge(1.0_16), -huge(1.0_16))) .EQ. zres(4)) r(6) = 1
  if (conjg((tiny(1.0_16), tiny(1.0_16))) .EQ. zres(5)) r(7) = 1
  if (conjg((epsilon(1.0_16), epsilon(1.0_16))) .EQ. zres(6)) r(8) = 1

  if (conjg(za1) .EQ. zres(1)) r(9) = 1
  if (conjg(za2) .EQ. zres(2)) r(10) = 1
  if (conjg(za3) .EQ. zres(3)) r(11) = 1
  if (conjg(za4) .EQ. zres(4)) r(12) = 1
  if (conjg(za5) .EQ. zres(5)) r(13) = 1
  if (conjg(za6) .EQ. zres(6)) r(14) = 1

  if (conjg(zb1) .EQ. zres(1)) r(15) = 1
  if (conjg(zb2) .EQ. zres(2)) r(16) = 1
  if (conjg(zb3) .EQ. zres(3)) r(17) = 1
  if (conjg(zb4) .EQ. zres(4)) r(18) = 1
  if (conjg(zb5) .EQ. zres(5)) r(19) = 1
  if (conjg(zb6) .EQ. zres(6)) r(20) = 1

  if (qconjg((1.0_16, 2.0_16)) .EQ. (1.0_16, -2.0_16)) r(21) = 1
  call check(r, e, 21)
end

!** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
!** See https://llvm.org/LICENSE.txt for license information.
!** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

!*  Test intrinsics function abs with quad precision complex.

program main
  use check_mod
  implicit none
  real*16, parameter :: maxr16 = sqrt(huge(1.0_16))
  real*16, parameter :: minr16 = tiny(1.0_16)
  !Maximum, Zero, Zero, Random number.
  complex(kind=16), parameter :: t1(1,2,1,1,2,1,1) = reshape(&
                    [(0.0_16, maxr16), (0.0_16, 0.0_16), (-0.0_16, -0.0_16),&
                     (-873245736712234.2348667234663246768734E-1000_16, &
                      -9.3958340233454645675235423234623532E-945_16)],&
                    (/1,2,1,1,2,1,1/))

  real(kind=16), parameter :: t2(4) = reshape(abs(t1), (/4/))
  real(kind=16) :: t3(4) = reshape(abs(t1), (/4/))
  real(kind=16), dimension(14) :: rslt, expect
  type mytype1
    complex(kind=16) :: m
  end type
  type(mytype1) :: dt1(2,1,1,1,1,1,3)

  !Maximum, Zero, Zero, Random number, Minimum.
  dt1(:,:,:,:,:,:,:)%m = reshape([(0.0_16, maxr16), (0.0_16, 0.0_16), (-0.0_16, -0.0_16),&
                                  (-873245736712234.2348667234663246768734E-1000_16, &
                                  -9.3958340233454645675235423234623532E-945_16),&
                                  (minr16, minr16), (0.0_16, minr16)], (/2,1,1,1,1,1,3/))
  expect = [1.09074813561941592946298424473378276E+2466_16, 0.0_16, 0.0_16,&
            9.39583402334546456752354232346235322E-0945_16,&
            1.09074813561941592946298424473378276E+2466_16, 0.0_16, 0.0_16,&
            9.39583402334546456752354232346235322E-0945_16,&
            1.09074813561941592946298424473378276E+2466_16, 0.0_16, 0.0_16,&
            9.39583402334546456752354232346235322E-0945_16,&
            4.75473186308633355902452847925549301E-4932_16,&
            3.36210314311209350626267781732175260E-4932_16]
  rslt(1:4) = t2
  rslt(5:8) = t3
  rslt(9:14) = reshape(abs(dt1(:,:,:,:,:,:,:)%m), (/6/))
  call checkr16(rslt, expect, 14, rtoler = 1.0E-33_16)
end program

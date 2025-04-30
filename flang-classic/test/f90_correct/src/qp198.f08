! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! Use option -Mq,10,2 to check ili of conversion from double to quad-precision
! and ili which represents a quad-precision function return value.

program test_ili
  real*8 :: r8 = 467345.45743_8
  real*16 :: r16
  r16 = qconv(r8)

contains
  function qconv(arg)
    real*8, intent(in) :: arg
    real*16, intent(out) :: qconv
    qconv = arg
  end function
end program

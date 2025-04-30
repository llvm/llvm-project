! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test FRACTION intrinsic with quad-precision arguments

program test
      integer :: i = 1
      integer :: exp_result(6), real_result(6)
      real(kind=16) :: b(6)
      real(kind=16) :: a(6)
      real(kind=16) :: x(6)

      exp_result = 1
      real_result = 0

      x(1) = 0.0_16
      x(2) = 3.0_16
      x(3) = 9654.2155_16
      x(4) = 45625.21578_16
      x(5) = 124870.123555_16
      x(6) = 45435462664325.8592342442351_16

      a(1) = 0.0000000000000000000000000000000000000000_16
      a(2) = 0.7500000000000000000000000000000000000000_16
      a(3) = 0.5892465515136718749999999999999999745777_16
      a(4) = 0.6961855435180664062499999999999999995994_16
      a(5) = 0.9526834377670288085937500000000000226798_16
      a(6) = 0.6456767588407197313563903421140821924802_16

      do while(i<7)
           b(i) = fraction(x(i))
           i = i+1
      end do

      if(a(1) == b(1)) real_result(1) = 1
      if(a(2) == b(2)) real_result(2) = 1
      if(a(3) == b(3)) real_result(3) = 1
      if(a(4) == b(4)) real_result(4) = 1
      if(a(5) == b(5)) real_result(5) = 1
      if(a(6) == b(6)) real_result(6) = 1
      call check(real_result, exp_result, 6)
end

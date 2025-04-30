!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
      program test
      integer, parameter :: num = 25
      integer results(num)
      integer, parameter :: expect(num)= (/32,0,2,32,0,2,16,0,0,11,5,32,1,4,32,5,0,32,32,32,26,32,30,27,32/)
      integer, parameter :: arr(5)=(/0,1,-108,4294901760,4294967295/)
      integer*8 var_test
     !1.a Use of constants
      results(1)=trailz(0)
      results(2)=trailz(1)
      results(3)=trailz(-108)
      results(4:8)=trailz(arr)
     !1.b Use of variables
      var_test=5_8
      results(9)=trailz(var_test)
      var_test= 2048_8
      results(10)=trailz(var_test)

     !1.d Chain use with trailz
      do i=11,15
          results(i)= trailz(trailz(arr(i-10)))
      end do
     !1.e Chain use of trailz and leadz
     !chain of trailz(leadz))
      do i=16,20
           results(i)=trailz(leadz(arr(i-15)))
      end do
     !chain of leadz(tarilz)
      do i=21,25
           results(i)=leadz(trailz(arr(i-20)))
      end do

      if (all( expect .eq. results)) then
          print *, 'expect  vs results match'
      else
          print *, 'expectz vs results  mismatch'
      endif

      call check(results, expect, num)
      end program

! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test that PRIVATE/PUBLIC can be used on names that aren't defined until later

module m1
 real max,min
 public:: max
 private:: min
 public:: sin, setmin
contains
 subroutine setmin(i)
  real i
  min = i
!print *,'min=',min
 end subroutine
 real function sin(x)
  real x
  sin = min*max*x
!print *,'sin=',sin
 end function
end module

program p
 use m1
 real t, u, v
 real expect(3), result(3)
 data expect/5.0,90.0,5.0/
 max = 9.0
 call setmin(2.0)
 t = 5.0
 u = sin(t)
 v = min(t,u)
 result(1) = t
 result(2) = u
 result(3) = v
! print *,t,u,v
 call check(result,expect,3)
end program

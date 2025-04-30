!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!

!	problems with data initialization
!	with modules and contained subprograms ... test all combinations

module m
 real x(5)
 data x/1,2,3,4,5/ !15

 contains

 subroutine s(results)
  real results(*)
  real y(5)
  data y/6,7,8,9,10/ !40
  results(1) = sum(x)
  results(2) = sum(y)
  call ss
 contains
  subroutine ss
   real z(5)
   data z/11,12,13,14,15/ !65
   r = sum(x+y+z)
   results(3) = sum(x)
   results(4) = sum(y)
   results(5) = sum(z)
  end subroutine
 end subroutine
end module

 use m
 real a(5)
 data a/16,17,18,19,20/ !90
 real results(13), expect(13)
 data expect /15,40,15,40,65,15,90,115,15,90,165,15,90/
 call s(results)
 call sub1
 call sub2
 results(12) = sum(x)
 results(13) = sum(a)

 !print *,results
 call check(results,expect,13)
contains
 subroutine sub1
  real b(5)
  data b/21,22,23,24,25/ !115
  results(6) = sum(x)
  results(7) = sum(a)
  results(8) = sum(b)
 end subroutine
 subroutine sub2
  real b(5)
  data b/31,32,33,34,35/ !165
  results(9) = sum(x)
  results(10) = sum(a)
  results(11) = sum(b)
 end subroutine
end

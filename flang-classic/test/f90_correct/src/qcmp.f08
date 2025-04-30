! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test comparison of quad-precision values

program main
  integer, parameter :: qp = 16
  real(qp) :: r16p, r16m
  logical:: rslt(12) = [1,1,1,1,1,1,1,1,1,1,1,1]
  logical:: expect(12) = [1,1,1,1,1,1,1,1,1,1,1,1] 
  r16p=1.1_16
  r16m=1.1_16

  if(r16p /= r16m) then
    rslt(1) = 0
  endif

  if(r16p /= 1.1_16) then
    rslt(2) = 0
  endif 

  r16m=0.1_16
  if(r16p < r16m) then
    rslt(3) = 0
  endif

  if(r16p < 01._16) then
    rslt(4) = 0
  endif 

  r16m=2.1_16
  if(r16p > r16m) then
    rslt(5) = 0
  endif 

  if(r16p > 2.1_16) then
    rslt(6) = 0
  endif 

  r16m=2.1_16
  if(r16p >= r16m) then 
    rslt(7) = 0
  endif

  if(r16p >= 2.1_16) then 
    rslt(8) = 0
  endif

  r16m=0.1_16
  if(r16p <= r16m) then
    rslt(9) = 0
  endif
	
  if(r16p <= 0.1_16) then
    rslt(10) = 0
  endif

  if(r16p == r16m) then 
    rslt(11) = 0
  endif 
	
  if(r16p == 0.1_16) then 
    rslt(12) = 0
  endif 

  call check(rslt, expect, 12)

end program main

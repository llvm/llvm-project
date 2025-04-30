! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! Target of a pointer assignment is a derived type array section with member.
! reported by a user in pgf90 3.0

module m
  type et
   integer:: v
   complex::xxyy
   double precision :: d(2)
   integer:: v2(2)
  end type
end module

  use m
  integer,pointer,dimension(:) :: p,q,r,s,t,u,w

  integer, dimension(37) :: results, expect
  data expect/1000,2000,3000,4000,5000,6000,7000,8000,9000,10000, &
             &1000,2000,3000,4000,5000,6000,7000,8000,9000,10000, &
             &2000,4000,6000,8000,10000,2000,4000,6000,8000,10000, &
             &2,4,6,8,10,4,8/


  type(et),dimension(10),target :: e

  do i = 1,10
   e(i)%v = 100*i
   e(i)%xxyy = (-1.0,-10.0)*i
   e(i)%d(1) = 99
   e(i)%d(2) = 98
   e(i)%v2(1) = 1000*i
   e(i)%v2(2) = i
  enddo

 5 format(a,i2,a,i2,a)
10 format( 10i6 )

  q => e%v2(1)
  !print 5,'q(',lbound(q,1),':',ubound(q,1),') => e%v2(1)      = (1000,2000,...,10000)'
  !print 10,q
  results(1:10) = q

  r => q
  !print 5,'r(',lbound(r,1),':',ubound(r,1),') => q           = (1000,2000,...,10000)'
  !print 10,r
  results(11:20) = r

  s => r(2:10:2)
  !print 5,'s(',lbound(s,1),':',ubound(s,1),') => r(2:10:2)   = (2000,4000,...,10000)'
  !print 10,s
  results(21:25) = s

  t => s(1:5:2)
  !print 5,'t(',lbound(t,1),':',ubound(t,1),') => s(1:5:2)    = (2000,6000,10000)'
  !print 10,t
  results(21:25) = s

  u => e(2:10:2)%v2(1)
  !print 5,'u(',lbound(u,1),':',ubound(u,1),') => e(1:10:2)%v2(2) = (2000,4000,6000,8000,10000)'
  !print 10,u
  results(26:30) = u

  w => e(2:10:2)%v2(2)
  !print 5,'w(',lbound(u,1),':',ubound(u,1),') => e(2:10:2)%v2(2) = (2,4,6,8,10)'
  !print 10,w
  results(31:35) = w

  p => w(2:5:2)
  !print 5,'p(',lbound(u,1),':',ubound(u,1),') => w(2:5:2)    = (4,8)'
  !print 10,p
  results(36:37) = p
  call check(results,expect,size(expect))
end

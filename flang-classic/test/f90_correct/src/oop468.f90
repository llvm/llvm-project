! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!

module mod
logical expect(31), rslt(31)
type :: objects(k1,k2,l1)
integer, kind :: k1 = 3
integer, kind :: k2 = selected_char_kind("ASCII")
integer, len :: l1
character(kind=k2,len=l1) :: c
integer  :: a(1+l1)
!integer   a(1+l1)
end type
contains
integer function check_dt(aaa)
type(objects(k2=1,k1=3,l1=:)),allocatable:: aaa
!type(objects(k2=1,k1=3,l1=*)):: aaa
rslt(26) = aaa%c .eq. 'abcd'
!print *, 'char = ',aaa%c
rslt(27) = size(aaa%a) .eq. 6
!print *, 'size = ',size(aaa%a)
rslt(28) = aaa%l1  .eq. 5
rslt(29) = len(aaa%c) .eq. 5
!print *, 'l1=',aaa%l1, len(aaa%c)
check_dt = len(aaa%c)
end function

end module

program p
use mod
integer i
type(objects(k2=1,k1=3,l1=4)) :: z
type(objects(k2=1,k1=3,l1=6)) :: y
type(objects(l1=:)),allocatable :: x

allocate(objects(k1=3,k2=1,l1=5) :: x)
rslt(1) = allocated(x)
!print *, 'l1=',x%l1
rslt(2) = x%l1 .eq. 5
do i=1,size(x%a)
x%a(i) = i
enddo
rslt(3) = x%l1 .eq. 5
rslt(4) = len(x%c) .eq. 5
!print *, 'l1=',x%l1, len(x%c)
do i=1,size(x%a)
rslt(4+i) = x%a(i) .eq. i
enddo
!print *, x%a

x%c = 'abcd'
y%c = 'stuv'
z%c = 'wxyz'


expect = .true.

rslt(11) = z%c .eq. 'wxyz'
rslt(12) = lbound(z%a,dim=1) .eq. 1
rslt(13) = ubound(z%a,dim=1) .eq. 5
rslt(14) = size(z%a) .eq. 5
!print *, z%c,lbound(z%a),ubound(z%a),size(z%a)
rslt(15) = x%c .eq. 'abcd'
rslt(16) = lbound(x%a,dim=1) .eq. 1
rslt(17) = ubound(x%a,dim=1) .eq. 6
rslt(18) = size(x%a) .eq. 6 
!print *, x%c,lbound(x%a),ubound(x%a),size(x%a)
rslt(19) = y%c .eq. 'stuv'
rslt(20) = lbound(y%a,dim=1) .eq. 1
rslt(21) = ubound(y%a,dim=1) .eq. 7
rslt(22) = size(y%a) .eq. 7 
!print *, y%c,lbound(y%a),ubound(y%a),size(y%a)

!print *, x%c, kind(x%c), len(x%c), 'l1=',x%l1
rslt(23) = x%c .eq. 'abcd'
rslt(24) = kind(x%c) .eq. 1
rslt(25) = x%l1 .eq. 5
i = check_dt(x)
rslt(30) = i .eq. 5
rslt(31) = len(x%c) .eq. 5
!print *, 'len =',i,len(x%c)

!print *, size(x%a)
!print *, lbound(x%a)
!print *, ubound(x%a)

call check(rslt,expect,31)

end

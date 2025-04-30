! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!

module mod
logical expect(8),rslt(8)
type :: stuff(k11,k22)
integer(k22) :: i
integer,kind :: k22 = 2 
integer,kind :: k11 = 4
integer(k22) :: j
end type
end module

program p
use mod


type(stuff) :: y

y = stuff(2,4)(i=3,j=4)

rslt(1) = y%i .eq. 3
rslt(5) = y%j .eq. 4
rslt(2) = kind(y%i) .eq. 2
rslt(3) = y%k11 .eq. 4
rslt(4) = y%k22 .eq. 2

expect = .true.
call check(rslt,expect,5)

end

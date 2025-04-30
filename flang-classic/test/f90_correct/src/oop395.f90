! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!

module mod
logical expect(2),rslt(2)
type :: stuff(k11,k22)
integer,kind :: k22 = 2
integer,kind :: k11
integer(k22) :: i
end type
type :: stuff2
integer :: i
end type

end module

program p
use mod


type(stuff(1,1)) :: y

rslt(1) = kind(y%i) .eq. 1
rslt(2) = y%k22 .eq. 1

expect = .true.
call check(rslt,expect,2)


end

!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!

program t
integer,pointer::xptr(:)
integer,target::x(1:5)
integer result,expect

x = 7
xptr=>x
!xptr=>x(2:4)
namelist/xx/ xptr

namelist/xx/ xptr
open(11, file='out', action='write')
write(11, '(A)') "    &xx xptr(3)=1/"
close(11)

open(17,file='out', action='read')
read(17, nml=xx)

expect=1
result=xptr(3)
call check(expect,result, 1)

end

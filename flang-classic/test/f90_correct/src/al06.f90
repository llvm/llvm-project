! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! tests errmsg clause

integer, allocatable, dimension(:) :: a
character(80) :: msg
integer :: status
integer, dimension(6) :: res
integer, dimension(6) :: exp = (/ 0, 0, 0, 0, 1, -1 /)

msg = ""
status = 99
allocate(a(1000), stat=status, errmsg=msg)
!print *, status, trim(msg)
res(1) = status
res(2) = (msg .ne. "")
deallocate(a, stat=status, errmsg=msg)
!print *, status, trim(msg)
res(3) = status
res(4) = (msg .ne. "")
deallocate(a, stat=status, errmsg=msg)
!print *, status, trim(msg)
res(5) = status
res(6) = (msg .ne. "")

call check(res, exp, 6)

end

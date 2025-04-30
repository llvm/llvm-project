!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! Checking for the token type "TK_G0FORMAT" (I0) and
! the special case of the token type "I" + "<integer>" (I02),
! while the normal case is I2.

program example
   integer :: i = 1, j = 2, k = 3
   character(len=2) :: expect(3) = (/'1 ', ' 2', ' 3'/)
   character(len=2) :: res

   write(res, 10) i
   if (res .ne. expect(1)) stop 1
   write(res, 20) j
   if (res .ne. expect(2)) stop 2
   write(res, 30) k
   if (res .ne. expect(3)) stop 3

   print *, 'PASS'

10 FORMAT (I0)
20 FORMAT (I02)
30 FORMAT (I2)
end


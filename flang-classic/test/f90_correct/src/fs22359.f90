! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

integer, PARAMETER :: AAA(3:4) = (/ -325, 400/)
integer , PARAMETER :: YYY(3:3) = (/ (abs((AAA(i))), i = 3,3) /)
integer , PARAMETER :: ZZZ = abs(aaa(3)) !!! works
integer pass

pass = 1;
if (zzz .eq. 325) then
  do i = 3, 3
    if (yyy(i) .ne. 325) then
      pass = 0
    end if
  end do
else
  pass = 0
endif

if (pass.eq.0) then
  print *, "FAIL"
else
  print *, "PASS"
endif

end

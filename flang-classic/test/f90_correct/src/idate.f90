! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

! idate requires uniform 2-byte or 4-byte arguments

integer*2 :: day2, month2, year2
integer*4 :: day4, month4, year4

character*4 :: result = "PASS"

call idate(month2, day2, year2)
call idate(month4, day4, year4)

write(*,'(3I5)') month2, day2, year2
write(*,'(3I5)') month4, day4, year4

if (day2   .lt. 1  .or. day2   .gt. 31) result = "fail"
if (month2 .lt. 1  .or. month2 .gt. 12) result = "fail"
if (year2  .lt. 18 .or. year2  .gt. 99) result = "fail"

if (day2   .ne. day4  ) result = "fail"
if (month2 .ne. month4) result = "fail"
if (year2  .ne. year4 ) result = "fail"

print*, result

end

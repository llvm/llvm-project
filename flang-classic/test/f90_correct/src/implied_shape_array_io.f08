! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! Test for IO with the whole implied-shape array.

program test
   implicit none
   integer :: i, length
   character(len = 6) :: str
   character(len = 12) :: str2
   character(len = 40) :: str3
   integer, parameter :: arr1(1:*) = (/(i, i = 1, 6)/)
   character, parameter :: arr2(1:*) = (/'a', 'b', 'c', 'd', 'e', 'f'/)
   integer, parameter :: arr3(1:*) = [10, 11, 12, 13, 14, 15]
   character, parameter :: arr4(1:*) = ['A', 'B', 'C', 'D', 'E', 'F']
   integer, parameter :: arr5(1:*,1:*) = reshape((/(i, i = 1, 20)/), (/ 4, 5 /))

   write(str, 10) arr1
   if (str .ne. '123456') stop 1
   write(str, 20) arr2
   if (str .ne. 'abcdef') stop 2
   write(str2, 30) arr3
   if (str2 .ne. '101112131415') stop 3
   write(str, 20) arr4
   if (str .ne. 'ABCDEF') stop 4
   write(str3, 40) arr5
   if (str3 .ne. ' 1 2 3 4 5 6 7 8 91011121314151617181920') stop 5

   inquire(IOLENGTH = length) arr1
   if (length .ne. 24) stop 6
   inquire(IOLENGTH = length) arr2
   if (length .ne. 6) stop 7
   inquire(IOLENGTH = length) arr3
   if (length .ne. 24) stop 8
   inquire(IOLENGTH = length) arr4
   if (length .ne. 6) stop 9
   inquire(IOLENGTH = length) arr5
   if (length .ne. 80) stop 10

   print *, 'PASS'
10 FORMAT (6I1)
20 FORMAT (6A1)
30 FORMAT (6I2)
40 FORMAT (20I2)
end

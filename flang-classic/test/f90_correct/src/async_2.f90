!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!

!
      program prog

      implicit none

          character, dimension(8), asynchronous :: a1
          logical rslt(8), expect(8)
          integer :: i

          ! Open a file, This case is asynchronous
          open(7,FORM='unformatted',FILE='hello1.txt',ASYNCHRONOUS='yes',ACTION='read',ACCESS='stream')
          open(8,FORM='unformatted',FILE='hello2.txt',ASYNCHRONOUS='yes',ACTION='read',ACCESS='stream')

!----------------------------

! -- Read from the abcd file: sync, then async, then sync.

          read(7,ASYNCHRONOUS='no') a1

          do i = 1,8
            rslt(i) = 'a' .eq. a1(i)
            expect(i) = .true.
          enddo

          call check(rslt, expect, 8)

          read(7,ASYNCHRONOUS='yes') a1
          wait(7)

          do i = 1,8
            rslt(i) = 'b' .eq. a1(i)
          enddo

          call check(rslt, expect, 8)

          read(7,ASYNCHRONOUS='no') a1

          do i = 1,8
            rslt(i) = 'c' .eq. a1(i)
          enddo

          call check(rslt, expect, 8)

! -- Read from the 0123 file: async, then sync, then async.

          read(8,ASYNCHRONOUS='yes') a1
          wait(8)

          do i = 1,8
            rslt(i) = '0' .eq. a1(i)
          enddo

          call check(rslt, expect, 8)

          read(8,ASYNCHRONOUS='no') a1
          
          do i = 1,8
            rslt(i) = '1' .eq. a1(i)
          enddo

          call check(rslt, expect, 8)

          read(8,ASYNCHRONOUS='yes') a1
          wait(8)

          do i = 1,8
            rslt(i) = '2' .eq. a1(i)
          enddo

          call check(rslt, expect, 8)

      end program prog


!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!

!

      program prog

      implicit none

          character, dimension(8), asynchronous :: a1
          character, dimension(16), asynchronous :: a2

          character, dimension(8), asynchronous :: b1
          character, dimension(16), asynchronous :: b2
          logical rslt(16), expect(16)
          integer :: i

	  ! Open a file, This case is asynchronous
          open(7,FORM='unformatted',FILE='hello1.txt',ASYNCHRONOUS='yes',ACTION='read',ACCESS='stream')
          open(8,FORM='unformatted',FILE='hello2.txt',ASYNCHRONOUS='yes',ACTION='read',ACCESS='stream')

!----------------------------

          read(7,ASYNCHRONOUS='yes') a1
          read(8,ASYNCHRONOUS='yes') a2
          wait(7)
          wait(8)

          do i = 1,8
            rslt(i) = 'a' .eq. a1(i)
            rslt(i+8) = '0' .eq. a2(i)
            expect(i) = .true.
            expect(i+8) = .true.
          enddo

          call check(rslt, expect, 16)

          do i = 1,8
            rslt(i) = '1' .eq. a2(i+8)
          enddo

          call check(rslt, expect, 8)

          read(7,ASYNCHRONOUS='yes') b1
          read(8,ASYNCHRONOUS='yes') b2
          wait(7)
          wait(8)

          do i = 1,8
            rslt(i) = 'b' .eq. b1(i)
            rslt(i+8) = '2' .eq. b2(i)
          enddo

          call check(rslt, expect, 16)

          do i = 1,8
            rslt(i) = '3' .eq. b2(i+8)
          enddo

          call check(rslt, expect, 8)

      end program prog

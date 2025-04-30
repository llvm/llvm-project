!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!

         function fooread(i)
         integer fooread,i,j
         open(12,file='fmtfile1')
         read(12, *) j
         fooread=i
!         print *, "i:",i,"j:",j
         if (j .ne. 9) then
             print *, "FAIL"
             stop
         end if
         end

         subroutine doread
         integer a,b,fooread
         integer c(3)
         character*20 writetome
         c(1) = 1
         c(2) = 2
         c(3) = 3
         a=3
         b=4
         open(11,file='fmtfile2')
         read(11, *) a,c(fooread(2)),b
!         print *, "read file2: a:1 c(fooread(2)), b", a, c(2), b
         if (a .eq. 1 .and. c(2) .eq. 1 .and. b .eq. 3) then
             print *, "READ OK"
         else
             print *, "FAIL"
             stop
         end if
         end

         function foo2(i)
         integer i,foo2
         write(10, fmt=102) i,i+2,i+3
102   format(i2,i2,i2)
         foo2 = i+4
         end

         function foo(i)
         integer foo,i,foo2
         open(12,file='fmtfile1')
         write(unit=12, fmt=111) i+7, foo2(1), i
         foo=i+3
111   format(i2, i2, i2)
         close(12)
         end

         program fmtreadwrite
         integer a,b,foo
         character*20 writetome
         a=1
         b=4
         open(10,file='fmtfile2')
         write(10, fmt=100) a,foo(2),b
100   format(i2,i2,i2)
         close(10)
         call doread()

         print *, "PASS"
         end



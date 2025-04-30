!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!



         function foounfr(i)
           integer foounfr,i,j
           open(12,file='unffile1.dat', form='unformatted')
           read(12) j
!            print *, "i, j:", i,j
           foounfr=i
         end

         subroutine unfread 
         integer a,b,foounfr
         integer c(3)
         character*20 writetome
         c(1) = 1
         c(2) = 2
         c(3) = 3
         a=3
         b=4
         open(10,file='unffile2.dat',form='unformatted')
         read(10) a,c(foounfr(2)),b
!          print *, "a c b", a, c, b
         if  (a .eq. 1 .and. c(1) .eq. 1 .and. c(2) .eq. 1 .and. & 	
             c(3) .eq. 3 .and. b .eq. 3) then
             print *, "PASS"
         else
             print *, "FAIL"
         endif
         end



         function foow2(i)
         integer i,foow2
         write(10) i,i+2,i+3
!            print *, "write to 10", i, i+2, i+3
         foow2 = i+1
         end

         function foow(i)
           integer foow,i,foow2
           open(12,file='unffile1.dat', form='unformatted')
           write(unit=12) i+7, foow2(1), i
!            print *, "write to 12", i+7, i+1, i
           foow=i+3
           close(12)
         end

         program mm
         integer a,b,foow
         character*20 writetome
         a=1
         b=4
         open(10,file='unffile2.dat', form='unformatted')
!          print *, "write to 10", a
         write(10) a,foow(1),b
!          print *, "write to 10", b
         close(10)

         call unfread()
         end

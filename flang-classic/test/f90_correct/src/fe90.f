C Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
C See https://llvm.org/LICENSE.txt for license information.
C SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
C
C PIC offset computed for zm(3) in line 30 is ignored in line 35


       program bidon
       call gllvls (66)
       stop
       end

       subroutine gllvls (nk)
       implicit none
       integer nk
       integer maxdynlv
       integer expect(3)
       integer result(3)
       parameter (maxdynlv = 1000)
       real zt(maxdynlv),ztr(maxdynlv),zm(maxdynlv)
       common /levelsr/ zt,ztr,zm
        integer k

       expect(1) = loc(zm(1))
       expect(2) = loc(zm(2))
       expect(3) = loc(zm(3))
C      print*, loc(zm(1)),loc(zm(2)),loc(zm(3))

       zm=0.
       ztr(1)=  zm(3)*0.5

C      print*, loc(zm(1)),loc(zm(2)),loc(zm(3))
       result(1) = loc(zm(1))
       result(2) = loc(zm(2))
       result(3) = loc(zm(3))
       call check(expect, result, 3)

         stop
       return
       end

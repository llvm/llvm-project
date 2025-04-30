** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
** See https://llvm.org/LICENSE.txt for license information.
** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

* Vectorizer - Loops should vectorize when the option -Mcray=pointer is used
*              even if the pointers are in common (llnl bug 32).

      program test32
      parameter(nn=3)
      integer result(nn), expect(nn)

      integer*4 i
      integer ynlbz(1000),ylloz(20,1000),ylhiz(20,1000)
      common /foo/ ynlbz,ylloz,ylhiz
      integer nlbz (1000),lloz(20,1000),lhiz(20,1000)
      pointer (xnlbz,nlbz)
      common /cnlbz/ xnlbz
      pointer (xlloz,lloz)
      common /clloz/ xlloz
      pointer (xlhiz,lhiz)
      common /clhiz/ xlhiz
      xnlbz = %loc(ynlbz)
      xlloz = %loc(ylloz)
      xlhiz = %loc(ylhiz)
      do i = 1, 1000
         nlbz(i) = 0
      enddo
      do i = 1, 1000
         do j = 1, 20
            lloz(j,i) = 0
            lhiz(j,i) = 0
         enddo
      enddo
      call blockset (19,22,23,1000)
      nerrs = 0
      do i = 1, 1000
         if (nlbz(i) .ne. 1) then
            nerrs = nerrs + 1
c            if (nerrs .le. 5) print *, 'nlbz: ',i, nlbz(i)
         endif
      enddo
      result(1) = nerrs

      nerrs = 0
      do j = 2, 19
         if (lloz(j,1) .ne. 23) then
            nerrs = nerrs + 1
c            if (nerrs .le. 5) print *, 'lloz: ',j,lloz(j,1)
         endif
      enddo
      result(2) = nerrs

      nerrs = 0
      do j = 2, 19
         if (lhiz(j,1) .ne. 22) then
            nerrs = nerrs + 1
c            if (nerrs .le. 5) print *, 'lhiz: ',j,lhiz(j,1)
         endif
      enddo
      result(3) = nerrs

      data expect/0, 0, 0/

      call check(result, expect, nn)

      end
c
c  the loops in this code will not vectorize as long as the pointers
c  are in common.  Can be gotten around with cpgi$ nodepchk, but shouldn't
c  need to do that.
c   compile with pgf77 -c -Mvect -Minfo=loop test32.f
      subroutine blockset(k1,l1,lmo,kmo)

      integer k, k1, l1, lmo, kmo

      pointer (xnlbz,nlbz(*))
      integer nlbz
      common  /cnlbz/ xnlbz

      pointer (xlloz,lloz(kmo,*))
      integer lloz
      common  /clloz/ xlloz

      pointer (xlhiz,lhiz(kmo,*))
      integer lhiz
      common  /clhiz/ xlhiz

      do 87000 k = 1 , kmo
        nlbz(k) = 1
87000 continue

      do 87002 k = 2 , k1
        lloz(k,1) = l1 + 1
        lhiz(k,1) = lmo - 1
87002 continue
 
 
      end

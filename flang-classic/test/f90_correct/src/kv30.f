C Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
C See https://llvm.org/LICENSE.txt for license information.
C SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
c
c Test for illegal loop interchange.  Originally found in MOLPRO.

      integer nshlr(11)
      integer nshl(11,8)
      integer expect
      data nshl / 0, 1, 9*0, 1, 0, 9*0, 66*0 /
      data nshell / 11 /
      data expect / 2 /
c
      nshlx=0
      nshlrx=0
      do 120 i=1,nshell
      if(nshlr(i).ne.0) nshlrx=i
      do 120 isy=1,8
      if(nshl(i,isy).ne.0) nshlx=i
120   continue
      call check(nshlx, expect, 1)
      end

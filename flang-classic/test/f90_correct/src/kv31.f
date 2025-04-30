c Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
c See https://llvm.org/LICENSE.txt for license information.
c SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
c
c Test case from MOLPRO.  The inner loop below should not be vectorized.

      double precision binom(100), expect(100)
      data expect /100 * 1.0d0/
      call cortab(binom, 10, 10)
      call checkd(binom, expect, 46)
      end

      subroutine cortab(binom,maxj,maxi)
      implicit double precision (a-h,o-z)
      dimension binom(*)
c
      inew=2
      binom(1)=1.0d0
      do 150 j=1,maxj
        do 140 i=1,j-1
c       do 140 i=1,j+1
          binom(inew)=binom(inew-1)
          inew=inew+1
  140   continue
  150 continue
      print *,inew
      end


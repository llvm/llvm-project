** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
** See https://llvm.org/LICENSE.txt for license information.
** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

*   Integer DO-loop tests - special cases:
*      (1) real addition used for step expression,
*      (2) do index variable used in init, upper, and step exprs,
*      (3) change values of init, upper, and step exprs
*          within the loop.

      program p
   
      integer rslts(11), expect(11)
c                         tests 1 - 4:
      data expect /   2, 7, 6, -4,
c                         tests 5 - 8:
     +                10, -6, -10, -8,
c                         tests 9 - 11:
     +                9, 1013, 11      /

C     tests 1, 3, 5, 2:   Real addition for step expression:

      data x, y / 5.3, -3.1 /
      do 10 i = 1, 6, x+y
          rslts(i) = 2 * i
10    continue
      rslts(2) = i

C     tests 4, 6, 8, 7:  DO index var used in DO expressions:

      data i6 /6/
      i = i6
      do 20 i = i-10, i-14, 2*i-14
          rslts(-i) = i
20    continue
      rslts(7) = i

C     tests 9, 11, 10:  Changing value of DO exprs within loop:

      j = 9
      k = 11
      m = 2
      do 30 i = j, k, m
          j = 100
          k = 1000
          m = 3
          rslts(i) = i
30    continue
      rslts(10) = i + k

C     check results:

      call check(rslts, expect, 11)
      end

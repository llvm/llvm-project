** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
** See https://llvm.org/LICENSE.txt for license information.
** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

*   Test integer DO loops (with CONTINUE as last statement).

      program p
      parameter (N = 26)
      integer rslts(N), expect(N)
      data rslts / N * 0 /
      data i1, in1 / 1, -1 /

c                                   tests 1 - 4:
      data expect / 2, 3, 4, 5,
c                                   tests 5 - 8:
     +              1, 1, 1, 1,
c                                   tests 9 - 11:
     +              10, 11, 33,
c                                   tests 12 - 14:
     +              -3, -1, -1,
c                                   tests 15 - 17:
     +              3, 3, 3,
c                                   tests 18 - 19:
     +              119, 120,
c                                   tests 20 - 24:
     +              0, 0, 0, 0, 7,
c                                   tests 25 - 26:
     +              9, 15   
     +            /

c   ------------------------- tests 1 - 4:

      do 10 i = 1, 4, i1
          rslts(i) = i + 1
10    continue

c   ------------------------- tests 5 - 8:

      do 20 i = 5, 7, 2
          do 30, j = i, i+1
              rslts(j) = rslts(j) + 1
   30     continue
20    continue

c   ------------------------- tests 9 - 11:

      x = 9.1
      do 40 j = x, 13.1D0, 1.1
          if (j .eq. 11)   goto 50
          rslts(j) = j + 1
40    continue
50    rslts(11) = j * 3

c   ------------------------- tests 12 - 14:

60    do 70, i = 14, 10, in1
          rslts(i) = -1
          if (i .eq. 12) then
              rslts(i) = rslts(i) - 2
              goto 80
          endif
70    continue

c   ------------------------- tests 15 - 17:

80    do 90 i = 15, 17, 1
          call f(rslts(i))
90    continue

c   ------------------------- tests 18 - 19:

      do 9999 i = i1, 1, i1
          do 9999 j = 18, 18 + i
              do 9999 k = j, j
                  rslts(j) = j + i + 100 + rslts(j)
9999  continue

c   ------------------------- tests 20 - 24:  zero trip loops

      do 2 i = 1, 0
          rslts(20) = 1
2     continue

      do 3, i = 7, i1
          rslts(21) = i
3     continue

      do 4 i = -3, i1, -2
          rslts(22) = i
4     continue

      do 5 i = 7, i1, i1
          rslts(23) = i
5     continue
      rslts(24) = i

c   ------------------------- tests 25 - 26:

      do 6 i = 1, 11, 7
          rslts(25) = rslts(25) + i
6     continue
      rslts(26) = i

c   ------------------------- check results:

      call check(rslts, expect, N)
      end

      subroutine f(i)

      do 10 j = -1, -100, -1
          i = i + 1
          if (j .eq. -3)  return
10    continue

      end

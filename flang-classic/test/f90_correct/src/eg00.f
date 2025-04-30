** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
** See https://llvm.org/LICENSE.txt for license information.
** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

*   DATA statements with implied DO loops.

      BLOCKDATA DO

      integer a(2, 3:4, -5:-4)
      real*4 b(4)
      integer * 2 c(4, 4), d(3,3), e(0:2, 0:2)
      character*2 ch(-3:2), g(1:3)*3
      double precision f(3)

      common /DO2/ i, a, b, c, d, e, ch, f, g

c  ----------------------------------------- tests 1 - 8:
c  initialize:        offset       value
c     a(1, 3, -5)  :    0           -1
c     a(2, 3, -5)  :    4           -2
c     a(1, 4, -5)  :    8           -5
c     a(2, 4, -5)  :   12           -6
c     a(1, 3, -4)  :   16           -3
c     a(2, 3, -4)  :   20           -4
c     a(1, 4, -4)  :   24           -7
c     a(2, 4, -4)  :   28           -8
      data i /-100/,
     '     (((a(i, j, k), i = 1, 1+1), k = -5, -4), j = 1*3, 4)/
     '     -1, -2, -3, -4, -5, -6, -7, -8 /

c  ----------------------------------------- tests 9 - 14:
c  initialize:
c    b(2)          :    4          7.0
c    b(4)          :   12          8.0
c    c(1, 1)       :    0            1
c    c(2, 2)       :   10            1
c    c(3, 3)       :   20            1
c    c(4, 4)       :   30            1
      data (b(i), i = 2, 5, 2) / 7, 8/,
     :     (c(i, i), i = 1, 4) / 4 * 1/

c  ------------------------------------------ tests 15 - 20:
c  initialize: 
c    d(1, 1)     :  0  : 2
c    d(1, 2)     :  6  : 2
c    d(2, 2)     :  8  : 4
c    d(1, 3)     : 12  : 4
c    d(2, 3)     : 14  : 4
c    d(3, 3)     : 16  : 4
      data ((d((i), j+i-i), i = 1, j), j = 1, 3) / 2*2, 4*4/

c  ------------------------------------------- tests 21 - 26:
c    ch(-3, -2, ... 1, 2)      '1 '  '2 '  '3 '  '3 '  '4 '  '55'
      data (ch(j),ch(j+1), j = -3,2,2) /'1','2',2*'3','4 ','55'/

c  ------------------------------------------- tests 27 - 33:
c  initialize:
c    e(0, 0)     :  0  : 1
c    e(1, 1)     :  8  : 2
c    e(2, 2)     : 16  : 3
c    e(0, 1)     :  6  : 4
c    e(0, 2)     : 12  : 5
c    e(2, 1)     : 10  : 6
c    e(1, 2)     : 14  : 7
      data (e(i-1,i-1), i=1,3), (e(i/i-1,i), i = 1, 2),
     +     (e(i*2, 1 + i - 1), i = 1, 1, 1), 
     +     (e(i/4, i/2+0), i = 4, 4, 99) 
     +  / 1, 2, 3, 4, 5, 6, 7/

c -------------------------------------------- test 34:
c    pad word between eresults and dresults.

c  ------------------------------------------- tests 35 - 40:
c  loop with negative step:
      data (f(i), i = 3, 1, -1) / 1.0D0, 2.0D0, 3.0D0/

c  ------------------------------------------- tests 41 - 43:
c  substring expression in implied do:
c  initialize last two bytes of g(1), g(2), g(3):

      data (g((i-1) * 2 - i)(2:3) , i = 3, 5) / '33', '44', '55' /

      end

C  ---- main program -----:

      integer a(8)
      real*4 b(4)
      integer * 2 c(4, 4), d(3,3), e(0:2, 0:2)
      character*2 ch(-3:2), g(1:3)*3
      double precision f(3)
      common /DO2/ i, a, b, c, d, e, ch, f, g

      parameter (N = 43)
      common/rslts/rslts(20),chrslts(6),erslts(8),drslts(3),grslts(3)
      integer          rslts, erslts
      character*4      chrslts, grslts
      double precision drslts

c ---- set up expected array:

      integer expect(N)
c           ---------------- tests 1 - 8:
      data expect / -1, -2, -5, -6, -3, -4, -7, -8,
c           ---------------- tests 9 - 14:
     +              7, 8, 1, 1, 1, 1,
c           ---------------- tests 15 - 20:
     +              2, 2, 4, 4, 4, 4,
c           ---------------- tests 21 - 26:
     +              '1   ', '2   ', '3   ', '3   ', '4   ', '55  ',
c           ---------------- tests 27 - 34:
     +              1, 2, 3, 4, 5, 6, 7, -99,
c           ---------------- tests 35 - 40: (3 d. p. values: 3.0, 2.0, 1.0):
c  BIG ENDIAN
c     +              '40080000'x, 0, '40000000'x, 0, '3ff00000'x, 0,
c  LITTLE ENDIAN
     +               0, '40080000'x, 0, '40000000'x, 0, '3ff00000'x,
c           ---------------- tests 41 - 43:
     +              '33  ', '44  ', '55  '           /

c ---- assign values to results array:

c  -- tests 1 - 8:
      do 10 i = 1, 8
10        rslts(i) = a(i)

c  -- tests 9 - 14:
      rslts(9) = b(2)
      rslts(10) = b(4)
      rslts(11) = c(1, 1)
      rslts(12) = c(2, 2)
      rslts(13) = c(3, 3)
      rslts(14) = c(4, 4)

c  -- tests 15 - 20:
      rslts(15) = d(1, 1)
      rslts(16) = d(1, 2)
      rslts(17) = d(2, 2)
      rslts(18) = d(1, 3)
      rslts(19) = d(2, 3)
      rslts(20) = d(3, 3)

c  -- tests 21 - 26:
      do 20 i = 1, 6
20        chrslts(i) = ch(i - 4)

c  -- tests 27 - 34:
      erslts(1) = e(0, 0)
      erslts(2) = e(1, 1)
      erslts(3) = e(2, 2)
      erslts(4) = e(0, 1)
      erslts(5) = e(0, 2)
      erslts(6) = e(2, 1)
      erslts(7) = e(1, 2)
      erslts(8) = -99

c  -- tests 35 - 40:
      drslts(1) = f(1)
      drslts(2) = f(2)
      drslts(3) = f(3)

c  -- tests 41 - 43:
      do 30 i = 1, 3
30        grslts(i) = g(i)(2:3)

c ---- check results:

      call check(rslts, expect, N)
      end                           

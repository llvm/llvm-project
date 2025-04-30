** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
** See https://llvm.org/LICENSE.txt for license information.
** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

      program P

*   Assignment statements - implicit type conversion of right hand
*   side to type of left hand side for integer, integer*2, real,
*   double precision, and complex.

      parameter (N = 46)
      integer   rslts(N),  iexpect(7), expect(N)
      integer*2 hrslts(8), hexpect(8)
      real      rrslts(7), rexpect(7)
      double precision drslts(7), dexpect(7)
      complex   crslts(7), cexpect(7)
 
      equivalence                            (iexpect, expect)
      equivalence (rslts( 8), hrslts(1)), (expect( 8), hexpect(1))
      equivalence (rslts(12), rrslts(1)), (expect(12), rexpect(1))
      equivalence (rslts(19), drslts(1)), (expect(19), dexpect(1))
      equivalence (rslts(33), crslts(1)), (expect(33), cexpect(1))

      integer i, j
      integer*2 h
      real x, y
      double precision d, e
      complex q, r

      data i, j  / 7, -3/
      data h     / -3   /
      data x, y  / -2.6, 5e4 /
      data d, e  / 2.11D1, -1.9D0 /
      data q, r  / (1.0, 2.0), (-3.1, 1.0) /

C  tests 1 - 7:  conversion of numeric types to INTEGER:

      rslts(1) = h
      rslts(2) = x
      rslts(3) = d
      rslts(4) = q
      rslts(5) = -2.6
      rslts(6) = 9D7
      rslts(7) = (3.1, 0.9)

      data iexpect / -3, -2, 21, 1, -2, 90000000, 3 /

C  tests 8 - 11:  conversion of numeric types to INTEGER*2:

      hrslts(1) = -i
      hrslts(2) = -x
      hrslts(3) = d
      hrslts(4) = q
      hrslts(5) = -23399
      hrslts(6) = 2.6
      hrslts(7) = 2.6D0
      hrslts(8) = (3.1, 0.9)

      data hexpect /-7, 2, 21, 1, -23399, 2, 2, 3 /

C  tests 12 - 18:  conversion of numeric types to REAL:

      rrslts(1) = j
      rrslts(2) = h
      rrslts(3) = -e
      rrslts(4) = r
      rrslts(5) = -23399
      rrslts(6) = 9D7
      rrslts(7) = (-3.1, 0.9)

      data rexpect / -3.0, -3.0, 1.9, -3.1, -23399.0, 9E7, -3.1/

C  tests 19 - 32:  conversion of numeric types to DOUBLE PRECISION:

      drslts(1) = -j
      drslts(2) = h
      drslts(3) = -x
      drslts(4) = -q
      drslts(5) = 23399
      drslts(6) = -2.0E6
      drslts(7) = (3.1, 0.9)

      data dexpect/3.0D0, -3.0D0, 2.6, -1.0D0, 23399D0, -2.0D6, 3.1/

C  tests 33 - 46:  conversion of numeric types to COMPLEX:

      crslts(1) = i
      crslts(2) = h
      crslts(3) = -y
      crslts(4) = -d
      crslts(5) = 3
      crslts(6) = 3.1
      crslts(7) = -9D7

      data cexpect/ (7.0, 0.0), (-3.0, 0.0), (-5e4, 0.0), (-21.1, 0.0),
     +              (3.0, 0.0), (3.1, 0.0),  (-9e7, 0.0)  /

C  ----  check results:

      call check(rslts, expect, N)

      end

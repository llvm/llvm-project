** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
** See https://llvm.org/LICENSE.txt for license information.
** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

*   Test implicit type conversions between numerical types
*   in DATA statements:

      BLOCK DATA
      implicit double precision (d-f)
      implicit complex (r-t)

      common /ic/ i, j, k, x, y, z, d, e, f, r , s, t

      data i, j, k / 2.3, 5D2, (3.2, 1.0) / ,
     =     x, y, z / 15, -3.2d-5, (-3.2, -1.0) / ,
     a     d, e, f / -3, 2.3, (3.2, 1.0) / ,
     Z     r, s, t / 99999, -2.3, 5d2    /
      end

      program p

      common /ic/ rslts(18)

      integer expect(18)
      real    rexpect(15)
      equivalence (expect(4), rexpect(1))
      double precision  dexpect(3)
      equivalence (expect(7), dexpect(1))

c  set up expected results for tests 1 - 3 (integer):

      data (expect(i), i = 1, 3) / 2, 500, 3/

c  set up expected results for tests 4 - 6 (real):

      data (rexpect(i), i = 1, 3) /15.0, -3.2e-5, -3.2 /

c  set up expected results for tests 7 - 12 (double):

      data x, y, z / -3.0e0, 2.3e0, 3.2e0 /
      dexpect(1) = x
      dexpect(2) = y
      dexpect(3) = z

c  set up expected results for tests 13 - 18 (complex):

      data (rexpect(i), i = 10, 15) /
     +        99999.0, 0.0,     -2.3, 0.0,     5e2, 0.0 /

c  compare results and expected arrays:

      call check(rslts, expect, 18)
      end

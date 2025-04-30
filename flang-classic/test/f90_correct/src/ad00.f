** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
** See https://llvm.org/LICENSE.txt for license information.
** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

*   32 bit hexadecimal and octal constants.

C   NONSTANDARD:
C     VMS notation for hex and octal constants.

      program p
      parameter (N = 'E'x)
      integer rslts(12), expect('14'o)
      logical lrslts(12), ll
      real    xrslts(N)
      equivalence (rslts, lrslts, xrslts)

      data expect / 0, 1, -1, 'ffffffff'x,
     +              0,15, -1, 'ABCDEFfa'x,
     +              8, 1, 0,  1           /

      rslts(1) = '0'o
      rslts(2) = '0'X + '1'O
      rslts(3) = 'fFfFfFfF'x
      rslts(4) = '  3777777777 7 'o

      rslts(5) = '0'o
      rslts(6) = (1 + '00'o) + N
      rslts(7) = - ('00000000'x) - 1
      rslts(8) = 'abcdefFA'X

      xrslts(9) = '10'O
      lrslts('a'x) = '1'x
      data x / 2.0 /
c ------ assume '40000000'x is bit representation of 2.0f:
      xrslts(11) = x - '40000000'x
      data ll / '0' x /
      lrslts(12) = ll .neqv. '1'o

      call check(rslts, expect, 'C'x)
      end

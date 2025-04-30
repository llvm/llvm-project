** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
** See https://llvm.org/LICENSE.txt for license information.
** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

*   Test data initialization of real, double, and
*   complex scalar variables.

      block data b

      real x, y, z
      double precision dx, dz
      real *8 dy
      real zz/10.2/
      common /s/ x, y, z, zz, dx, dy, dz, c1, c2, c3
      complex c1, c2, c3

      data x, y /0.0, 2.3e0/,  z / -5.6E20 /
      data dx, dy, dz / -0D0, 5d1, -4.0D-3 /
      data c1/(0.1, 1.0)/
     +     c2/(-1.0, 0.0)/
     +     c3/-(2.3, -4.5)/

      END

C   MAIN PROGRAM -
      common /s/ results(16)
      parameter (x = 0.0)
      real expect(16)
      double precision d(2)
      equivalence (expect(7), d(1))
      
      data expect /
c            tests 1 - 4:  real numbers
     +                     0.0, 2.3, -5.6E20, 10.2,
c            tests 5 - 10: double precision 
     +                     0.0, -0.0, x, x, x, x,
c            tests 11 - 16: complex
     +                     0.1, 1.0, -1.0, 0.0, -2.3, 4.5 /

      d(1) = 50.0D0
      d(2) = -4.0D-3

      call  check(results, expect, 16)
      end

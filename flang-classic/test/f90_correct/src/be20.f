** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
** See https://llvm.org/LICENSE.txt for license information.
** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

*   EQUIVALENCE statements - data dependencies

      program p
      parameter (n=3)
      integer results(n)
      integer expect(n)

c   the variables in the following equivalences cannot be used as
c   arguments so that they will not have their addtkn flag set.
c   this gives the register allocator the opportunity to rule them
c   out as candidates due to their "equivalenced" attribute.

      EQUIVALENCE (RVOE01, RVOE02)
      EQUIVALENCE (IVOE12, IVOE13), (IVOE13, IVOE14)
      EQUIVALENCE (IVOE15, IVOE16)
      EQUIVALENCE (IVOE16, IVOE17)

      RVCOMP = 0.0
      RVOE01 = 45.
      RVOE02 = 12.
      RVCORR = 12.
      RVCOMP = RVOE01
      call fset(results(1), rvcomp)	! test 1

      IVCOMP = 0
      IVOE12 = 12
      IVOE13 = 13
      IVOE14 = 14
      IVCORR = 14
      IVCOMP = IVOE13
      call iset(results(2), ivcomp)	! test 2

      IVCOMP = 0
      IVOE15 = 15
      IVOE16 = 16
      IVOE17 = 17
      IVCORR = 17
      IVCOMP = IVOE16
      call iset(results(3), ivcomp)	! test 3

      call check(results, expect, n)

	data  expect /12, 14, 17/

      end
      subroutine fset (i, x)	! integer <- single
      i = x
      end
      subroutine iset (i, j)	! integer <- integer
      i = j
      end

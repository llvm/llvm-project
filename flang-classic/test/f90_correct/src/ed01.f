** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
** See https://llvm.org/LICENSE.txt for license information.
** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

*   Data initialization using Hollerith constants and multi-word
*   variables.

c  Ensures that when a multi-word variable is traversed as a "char *",
c  the lexical order of the hollerith constant is maintained.


      program p
      parameter (N=32)
      integer rslts(N), expect(N)

      complex cp(1)
      double complex dcp(1)
      double precision dp(1)
      data cp/8habcdefgh/
      data dp/8habcdefgh/
      data dcp/16habcdefgh12345678/

      data expect /
     + 97,98,99,100,101,102,103,104,  !cp
     + 97,98,99,100,101,102,103,104,  !dp
     + 97,98,99,100,101,102,103,104,49,50,51,52,53,54,55,56 !dcp
     +/

      call fill(rslts(1), cp, 8)
      call fill(rslts(9), dp, 8)
      call fill(rslts(17), dcp, 16)

      call check(rslts, expect, N)
      end
      subroutine fill(r, bt, n)
      integer r(*)
      byte bt(*)
      do i=1, n
	r(i) = bt(i)
      enddo
      end

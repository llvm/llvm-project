
** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
** See https://llvm.org/LICENSE.txt for license information.
** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

**--- BZ edit descriptor

      parameter (N=2)
      common ires(N), iexp(N)
      ires(1) = itest1()	! leading blanks
      ires(2) = itest2()	! leading and embedded blanks

      data iexp/-1, -1010/
      call check(ires, iexp, N)
      end
      integer function itest1()
      character *5 ch
      data ch /'  -1'/
      read (ch, 100) itest1
 100  format (BZ,I4)
      return
      end
      integer function itest2()
      character *6 ch
      data ch /' -1 1 '/
      read (ch, 100) itest2
 100  format (BZ,I6)
      return
      end

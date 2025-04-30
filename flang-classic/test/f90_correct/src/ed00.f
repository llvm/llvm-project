** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
** See https://llvm.org/LICENSE.txt for license information.
** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

*   Data initialization using Hollerith constants:

      blockdata h
      common /h2/ i, j, k, n, x, y, d, si, ll, sll, ch
      double precision d
      integer*2 si
      logical ll
      logical*1 sll
      character ch*27

      data i, j, k, n / 4Habcd, 1he, 5Hfghij, 'klmn' /
      data x, y, d    / 4hopqr, 2hst, 8h12345678     /
      data si, ll, sll / 'ab', 3Hcde, 'f'            /
      data ch(1:1), ch(2:27) /1ha, 27hbcdefghijklmnopqrstuvwxyz/-/
      end

      program p
      common /h2/ rslts
      character*4 rslts(17), expect(17)
      integer*4 irslts(17), iexpect(17)
      equivalence (irslts, rslts)
      equivalence (iexpect, expect)

      integer *2 i
      equivalence (i, rslts(9)(3:4))

      data expect / 'abcd', 'e   ', 'fghi', 'klmn',
     +              'opqr', 'st  ', '1234', '5678',
     +              'ab..', 'cde ', 'fabc', 'defg',
     +              'hijk', 'lmno', 'pqrs', 'tuvw',
     +              'xyz/'  /

c     fill in 2 bytes in /h/ which would otherwise be undefined:
c     BIG ENDIAN
c      i = 4h  ..
c     LITTLE ENDIAN
      i = 4h..

      call check(irslts, iexpect, 17)
      end

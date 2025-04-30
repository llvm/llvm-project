** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
** See https://llvm.org/LICENSE.txt for license information.
** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

*   Data initialization of character scalars and substrings.

      common /character2/ rslts
      character * 4 rslts(11), expect(11)
      integer * 4 irslts(11), iexpect(11)
      equivalence (irslts, rslts)
      equivalence (iexpect, expect)
      data expect / 'a''\01z',   ' wxy',
     +              'stuv',      'lmno',
     +              'pa  ',      'bc  ',
     +              '    ',      '    ',
     +              'defc',      'zwab',
     +              'xm j'                /

      call check(irslts, iexpect, 11)
      end

      block data character

      character * 3 c1 *1, d1 *1, d3, c3, e3, f3
      character     e1, c2 * 2, f1, e4*4, f5*5
      character     f12*12, a8*(8)

      common /character2/ c1, d1, e1, c2, d3, e4, f5, c3, f12, e3, f1
      common /character2/ a8

      data c1/'a'/  d1/''''/, e1/'\01'/
      data c2, d3, e4, f5 / 'z ', 'wxy', 'stuv', 'lmnop' /
c   initializations which require padding of character strings:
      data c3, f12 / 'a', 'bc' /
c   initializations which require truncation of character strings:
      data e3, f1 / 'def ', 'cowpie' /

c   initialization of substrings: set a8 to 'zwabxm j':

      data a8(:1), a8(2:2), a8(5:5), a8(3:4), a8(6:7), a8(8:) /
     -     'z',    'wy',    'x',     'ab',    'm',     'j'    /
      end


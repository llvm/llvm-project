** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
** See https://llvm.org/LICENSE.txt for license information.
** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

*   Character string constants.

      program p
      parameter(n = 30)
      integer rslts(n), expect(n)

      character ch*14, ch2*10

c  ---- set up expected array:

      data expect / 10, 9, 8, 12, 13, 0, 39,
     +              34, 92, 97, 255, 34, 39, 65,
     +              10, 39, 39, 34, 34,
     +              65, 1, 56, 47, 44,
     +              203, 1, 1, 6, 8, 2           /

c  ---- tests 1 - 14:

      data ch /'\n\t\b\f\r\0\'\"\\\a\377"''A'/

      do 10 i = 1, 14
10        rslts(i) = ichar( ch(i:i) )

c  ---- tests 15 - 24:

      data ch2 / "\n''""""A\001\070/," /

      do 20 i = 1, 10
20        rslts(14 + i) = ichar( ch2(i:i) )

c  ---- tests 25 - 30:

      rslts(25) = LEN('456789 123456789 123456789 123456789  23456789 1
     +
     + 456789        89 123456789 123456789 123456789 123456789 12345
     +9 123456789 123456789 ')

      rslts(26) = LEN(' ')
      rslts(27) = LEN('''')
      rslts(28) = ichar( '\6' )
      rslts(29) = ichar('\10')
      rslts(30) = len( '\0123' )

c  ---- check results:

      call check(rslts, expect, n)
      end 

** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
** See https://llvm.org/LICENSE.txt for license information.
** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

*   Miscellaneous Scanner tests.

C  The following items are tested:
C   (1) 31 character identifiers, special characters in identifiers.
C   (2) id characters after 63 ignored.
C   (3) variable named doi10 and array named IF.
C   (4) illegal characters should be ignored with warning msg.
C   (5) token sequence which looks almost like complex constant.
C   (6) ''                     ''              real or hollerith constant.
C   (7) Hollerith constant which extends to last character of line.

C   NONSTANDARD:
C     Use of $ _ in identifier names (VMS).
C     Identifiers beginning with $ or _ (non-VMS).
C     Illegal characters - semicolon, backslash - warning should be issued.
C
C     As of 10/23/2012, this test should be compiled with -Mstandard 

      program a234567890123456789012345678901xXxXxXxXxXxXxXxXxXxX
      implicit integer ($, d)
      parameter( N = 14 )
      integer _, __, character
      parameter($=3, __ = 4)
      dimension if(3)
      character * 2e3
      character * 4habcd
      integer rslts(n), expect(n)

      data expect / 9, 201, 7, 8, 3, 4, 8, -2,
     +              3, 1, 3, 10, 2000, 2       /

      H = 2.3
      rslts(1) = 4*h
      o23456789012345678901234567890123456789012345678901234567890123y =
     +0
      o23456789012345678901234567890123456789012345678901234567890123x =
     +100.9
      rslts(2) =
     +o23456789012345678901234567890123456789012345678901234567890123y*2

      call z23_$67890123456789012345678901( rslts(3) )
      call z23_$67890123456789012345678902( rslts(4) )
      rslts(5) = $
      rslts(6) = __

      $$ = 2.3
      rslts(7) = $$*4
      data_/-2/
      rslts(8) = _

      do 10 i = 3
      rslts(9) = do10i
      DO 10 i = (1.1, 2.0)
      rslts(10) = DO 10 i

      i = 1
      if(i) = 3
      rslts(11) = IF(i)

      ;rslts\(12) = 10;

      if (i .ne. 1) then
           write(10,10) (2.3, 2.3, i = 1, 2)
 1 0  format(56H  ...
     +)
          write(10,10) (2.3, 2.3)
      end if

      character = 1
      rslts(13) = character * 2e3
      i1h = 2
      rslts(14) = i1h

C  ----------check results:
  
      call check(rslts, expect, N)
      end
      subroutinez23_$67890123456789012345678901( i )
      i = 7
      return
      entry z23_$67890123456789012345678902( i )
      i = 8
      end

!* Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
!* See https://llvm.org/LICENSE.txt for license information.
!* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

!   Miscellaneous Scanner tests.
!     freeform version of ag00

!  The following items are tested:
!   (1) 31 character identifiers, special characters in identifiers.
!   (2) id characters after 63 ignored.
!   (3) variable named doi10 and array named IF.
!   (4) illegal characters should be ignored with warning msg.
!   (5) token sequence which looks almost like complex constant.
!   (6) ''                     ''              real or hollerith constant.
!   (7) Hollerith constant which extends to last character of line.

!   NONSTANDARD:
!     Use of $ _ in identifier names (VMS).
!     Identifiers beginning with $ or _ (non-VMS).
!     Illegal characters - backslash - warning should be issued.
!
!     As of 10/23/2012, this test should be compiled with -Mstandard


      program a234567890123456789012345678901xXxXxXxXxXxXxXxXxXxX
      implicit integer ($, d)
      parameter( N = 14 )
      integer _, __, character
      parameter($=3, __ = 4)
      dimension if(3)
      character *2 e3
      character *4 habcd
      integer rslts(n), expect(n)

      data expect / 9, 201, 7, 8, 3, 4, 8, -2,        &
     &              3, 1, 3, 10, 2000, 2       /

      H = 2.3
      rslts(1) = 4*h
      o23456789012345678901234567890123456789012345678901234567890123y = 0
      o23456789012345678901234567890123456789012345678901234567890123x = 100.9
      rslts(2) = o23456789012345678901234567890123456789012345678901234567890123y*2

      call z23_$67890123456789012345678901( rslts(3) )
      call z23_$67890123456789012345678902( rslts(4) )
      rslts(5) = $
      rslts(6) = __

      $$ = 2.3
      rslts(7) = $$*4
      data _/-2/
      rslts(8) = _

      do10i = 3
      rslts(9) = do10i
      DO10i = (1.1, 2.0)
      rslts(10) = DO10i

      i = 1
      if(i) = 3
      rslts(11) = IF(i)

      ;rslts\(12) = 10;

      if (i .ne. 1) then
           write(10,10) (2.3, 2.3, i = 1, 2)
 10  format(56H  ...                                                   &
     &)
          write(10,10) (2.3, 2.3)
      end if

      character = 1
      rslts(13) = character * 2e3
      i1h = 2
      rslts(14) = i1h

!  ----------check results:
  
      call check(rslts, expect, N)
      end
      subroutine z23_$67890123456789012345678901( i )
      i = 7
      return
      entry z23_$67890123456789012345678902( i )
      i = 8
      end

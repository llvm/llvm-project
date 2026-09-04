! RUN: not %flang_fc1 -fsyntax-only -pedantic -Werror %s 2>&1 | FileCheck %s --check-prefix=ERROR

      implicit none
      integer :: i

! ERROR: fixed-c-comment02.F:9:1: portability: nonstandard usage: C-style comment
! ERROR: fixed-c-comment02.F:10:3: error: Character in fixed-form label field must be a digit
      i = 11
/*
*/+1

! ERROR: fixed-c-comment02.F:13:1: portability: nonstandard usage: C-style comment
/* Single-line C comment */
! ERROR: fixed-c-comment02.F:15:1: portability: nonstandard usage: C-style comment
/* Multi-line
 * C comment
 */

! ERROR: fixed-c-comment02.F:22:1: portability: nonstandard usage: C-style comment
! ERROR: fixed-c-comment02.F:22:6: error: unterminated C-style comment
      i = 21
/**/ /* 2
! ERROR: fixed-c-comment02.F:25:6: error: unterminated C-style comment
      i = 22
     /* 2
! ERROR: fixed-c-comment02.F:29:1: error: unterminated C-style comment
! ERROR: fixed-c-comment02.F:29:1: error: Character in fixed-form label field must be a digit
! ERROR: fixed-c-comment02.F:29:1: portability: nonstandard usage: C-style comment
/* Not-terminated C comment

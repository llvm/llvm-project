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

! ERROR: fixed-c-comment02.F:23:6: portability: nonstandard usage: C-style comment
! ERROR: fixed-c-comment02.F:25:29: error: Incomplete character literal
      i = 21
      i = i
     /* 2
      print *, i
      print *, 'tail */ text'

! ERROR: fixed-c-comment02.F:30:1: portability: nonstandard usage: C-style comment
! ERROR: fixed-c-comment02.F:30:6: error: unterminated C-style comment
      i = 31
/**/ /* 2
! ERROR: fixed-c-comment02.F:33:6: error: unterminated C-style comment
      i = 32
     /* 2
! ERROR: fixed-c-comment02.F:37:1: error: unterminated C-style comment
! ERROR: fixed-c-comment02.F:37:1: error: Character in fixed-form label field must be a digit
! ERROR: fixed-c-comment02.F:37:1: portability: nonstandard usage: C-style comment
/* Not-terminated C comment
      end

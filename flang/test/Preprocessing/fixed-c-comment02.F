! RUN: not %flang_fc1 -fsyntax-only -pedantic -Werror %s 2>&1 | FileCheck %s --check-prefix=ERROR
! RUN: not %flang_fc1 -nocpp -fsyntax-only -pedantic -Werror %s 2>&1 | \
! RUN:    FileCheck %s --check-prefix=NOCPP-ERROR

      implicit none
      integer :: i

! ERROR: fixed-c-comment02.F:11:1: portability: nonstandard usage: C-style comment
! ERROR: fixed-c-comment02.F:12:3: error: Character in fixed-form label field must be a digit
      i = 11
/*
*/+1

! ERROR: fixed-c-comment02.F:16:1: portability: nonstandard usage: C-style comment
! NOCPP-ERROR: fixed-c-comment02.F:16:1: error: Character in fixed-form label field must be a digit
/* Single-line C comment */
! ERROR: fixed-c-comment02.F:18:1: portability: nonstandard usage: C-style comment
/* Multi-line
 * C comment
 */

! ERROR: fixed-c-comment02.F:26:6: portability: nonstandard usage: C-style comment
! ERROR: fixed-c-comment02.F:28:29: error: Incomplete character literal
      i = 21
      i = i
     /* 2
      print *, i
      print *, 'tail */ text'

! ERROR: fixed-c-comment02.F:33:1: portability: nonstandard usage: C-style comment
! ERROR: fixed-c-comment02.F:33:6: error: unterminated C-style comment
      i = 31
/**/ /* 2
! ERROR: fixed-c-comment02.F:36:6: error: unterminated C-style comment
      i = 32
     /* 2
! ERROR: fixed-c-comment02.F:39:1: error: unterminated C-style comment
! ERROR: fixed-c-comment02.F:39:1: error: Character in fixed-form label field must be a digit
/* Not-terminated C comment
      end

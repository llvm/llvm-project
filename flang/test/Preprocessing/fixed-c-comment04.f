! RUN: not %flang_fc1 -cpp -fsyntax-only %s 2>&1 | FileCheck %s --check-prefix=ERROR

      implicit none
      integer :: x, y
      y = 3
! The lines below should be parsed as distinct lines, with no continuation.
! ERROR: error: obsolete legacy extension is not supported
      x = y
	/**/+2
      print *, x
      end

! RUN: not %flang_fc1 -fsyntax-only %s 2>&1 | FileCheck %s --check-prefix=ERROR

      integer :: i
! ERROR: error: expected end of statement
      i = 8 /* comment */
      end

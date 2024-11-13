! RUN: %flang_fc1 -fdebug-unparse %s 2>&1 | FileCheck %s
subroutine subr(ar)
  real(8) :: ar(..)
!CHECK:  PRINT *, 8_4
  print *, kind(ar)
end

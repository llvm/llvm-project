! RUN: %flang_fc1 -fdebug-unparse %s 2>&1 | FileCheck %s
subroutine sub3(ar_at)
  type(*) :: ar_at(..)
!CHECK:  PRINT *, int(int(rank(ar_at),kind=8),kind=4)
  print *, rank(ar_at)
end

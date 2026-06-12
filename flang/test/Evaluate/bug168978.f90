!RUN: %flang_fc1 -fdebug-unparse %s 2>&1 | FileCheck %s
subroutine sub(dd)
  type(*)::dd(..)
  !CHECK: PRINT *, size(lbound(dd))
  print *, size(lbound(dd)) ! do not fold
end

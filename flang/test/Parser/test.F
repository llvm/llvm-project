! RUN: %flang_fc1 -fdebug-unparse-no-sema %s 2>&1 | FileCheck %s

! CHECK: SUBROUTINE foo
! CHECK: INTEGER i
! CHECK: i = 1
! CHECK: PRINT *, i
! CHECK: END SUBROUTINE foo

subroutine foo()
#include "inc.h"
  integer :: i
  i = VAL
  print *, i
end subroutine foo

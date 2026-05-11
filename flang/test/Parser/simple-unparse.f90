! Test that SIMPLE function specifier is recognized
! by the parser and the unparser. This test does not
! exercise semantic checks.
! RUN: %flang_fc1 -fdebug-unparse-no-sema %s 2>&1 | FileCheck %s

simple function foo()
  return
end function

! CHECK: SIMPLE FUNCTION foo()
! CHECK-NEXT: RETURN
! CHECK-NEXT: END FUNCTION


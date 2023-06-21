! RUN: %flang_fc1 -fdebug-unparse %s 2>&1 | FileCheck %s
! Continuation right before character literal.
subroutine test()
! CHECK: CHARACTER(LEN=3_4) :: a = "ABC"
  character(len=3) :: a =&
"ABC"
end subroutine

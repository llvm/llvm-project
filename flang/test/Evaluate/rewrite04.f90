! RUN: %flang_fc1 -fdebug-unparse %s 2>&1 | FileCheck %s
! Ensure folding of 1*j is a parenthesized (j) when j is a variable.
call foo(1*j)
!CHECK: CALL foo((j))
end

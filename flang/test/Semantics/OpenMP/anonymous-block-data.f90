!RUN: %flang_fc1 -fdebug-unparse -fopenmp %s | FileCheck %s

! To trigger the crash, -fsyntax-only was sufficient, but when everything
! is correct, it won't produce any output. To get something to check on
! success, run unparse, which does run semantic checks.

block data
end

!CHECK: BLOCK DATA
!CHECK: END BLOCK DATA

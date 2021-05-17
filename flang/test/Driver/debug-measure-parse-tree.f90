! Ensure argument -fdebug-measure-parse-tree works as expected.

!----------
! RUN LINE
!----------
! RUN: %flang_fc1 -fsyntax-only -fdebug-measure-parse-tree %s  2>&1 | FileCheck %s --check-prefix=FRONTEND

!-----------------
! EXPECTED OUTPUT
!-----------------
! FRONTEND:Parse tree comprises {{[0-9]+}} objects and occupies {{[0-9]+}} total bytes.

program A
end

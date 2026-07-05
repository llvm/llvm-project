! RUN: not %flang_fc1 -fsyntax-only %s 2>&1 | FileCheck %s
!CHECK: expected 'THEN'
!CHECK-NOT: expected 'PAUSE'
if (.TRUE.)
!CHECK: expected 'THEN'
else if (.FALSE.)
end if
end

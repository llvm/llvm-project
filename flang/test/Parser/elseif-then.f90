! RUN: not %flang_fc1 -fsyntax-only %s 2>&1 | FileCheck %s
! CHECK-NOT: expected '=>'
! CHECK: error: expected 'THEN'
if (.false.) then
else if (.false.)
else
end if
end

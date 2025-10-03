! RUN: not %flang_fc1 -fsyntax-only %s 2>&1 | FileCheck %s
continue
! CHECK: error: expected end of statement
flush iostat=1
end

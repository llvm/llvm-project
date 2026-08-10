! RUN: not %flang_fc1 -fsyntax-only %s 2>&1 | FileCheck %s
! CHECK: 3:8: error: invalid space
x = 1._ 4
end

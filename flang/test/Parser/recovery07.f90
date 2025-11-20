! RUN: not %flang_fc1 -fsyntax-only %s 2>&1 | FileCheck %s
! CHECK: error: expected ':'
! CHECK: in the context: loop control
do concurrent(I = 1, N)
end do
end

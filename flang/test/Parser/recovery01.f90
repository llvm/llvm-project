! RUN: not %flang_fc1 -fsyntax-only %s 2>&1 | FileCheck %s
program main
    call foo(i, &
             j, &
             k, &
             1$)
end

!CHECK: error: expected ')'
!CHECK: in the context: CALL statement

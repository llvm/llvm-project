! RUN: %flang_fc1 -fsyntax-only %s 2>&1 | FileCheck %s
!CHECK: warning: Compiler directive was ignored
!DIR$ Not a recognized directive
end

! RUN: %flang_fc1 -fsyntax-only %s 2>&1 | FileCheck %s
!CHECK: warning: Unrecognized compiler directive was ignored
!DIR$ Not a recognized directive
program main
 contains
  !CHECK: warning: Compiler directive ignored here
  !DIR$ not in a subprogram
  subroutine s
  end
end

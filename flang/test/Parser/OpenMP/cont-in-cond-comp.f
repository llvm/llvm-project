! RUN: not %flang_fc1 -fopenmp -fsyntax-only %s 2>&1 | FileCheck %s
! CHECK: warning: Statement should not begin with a continuation line [-Wscanning]
c$   !   0      !

      print *,'pass'
      end

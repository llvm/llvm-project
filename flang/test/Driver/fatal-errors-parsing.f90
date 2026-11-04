!RUN: not %flang_fc1 -fsyntax-only -Wfatal-errors %s 2>&1 | FileCheck %s --check-prefix=CHECK1
!RUN: not %flang_fc1 -fsyntax-only %s 2>&1 | FileCheck %s --check-prefix=CHECK2

program p
 contains
   ! CHECK1: fatal-errors-parsing.f90:{{.*}} error:
   ! CHECK2: fatal-errors-parsing.f90:{{.*}} error:
continue
end

subroutine s
contains
   ! CHECK1-NOT: error:
   ! CHECK2: fatal-errors-parsing.f90:{{.*}} error:
continue
end

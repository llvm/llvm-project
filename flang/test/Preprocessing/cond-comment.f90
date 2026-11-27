!RUN: %flang_fc1 -fopenmp -fdebug-unparse %s 2>&1 | FileCheck %s
!CHECK: END
!CHECK-NOT: error:
end
!$ !

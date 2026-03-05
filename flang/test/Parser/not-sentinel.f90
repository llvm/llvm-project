!RUN: flang -fc1 -fopenmp -fdebug-unparse %s 2>&1 | FileCheck %s
!CHECK-NOT: error:
!CHECK: END PROGRAM
!$ompx foo
end

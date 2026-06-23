! RUN: bbc -fopenacc -emit-hlfir %s -o - 2>&1 | FileCheck %s

module acc_routine_bad

contains
!$acc routine
subroutine acc_routine20()
end subroutine
end module

!CHECK: {{.*}}warning: Compiler directive ignored here{{.*}}

! RUN: bbc -fopenacc -emit-hlfir %s -o - 2>&1 | FileCheck %s

module acc_routine_bad

contains
!$acc routine
subroutine acc_routine20()
end subroutine
end module

!CHECK: {{.*}}warning: OpenACC routine directive without name must be placed in a subroutine or function

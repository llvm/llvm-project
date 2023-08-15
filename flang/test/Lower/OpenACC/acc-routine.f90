! This test checks lowering of OpenACC routine directive.

! RUN: bbc -fopenacc -emit-fir %s -o - | FileCheck %s

subroutine acc_routine1()
  !$acc routine
end subroutine

! CHECK: acc.routine @acc_routine_0 func(@_QPacc_routine1)
! CHECK-LABEL: func.func @_QPacc_routine1()

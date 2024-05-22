! This test checks lowering of OpenACC routine directive.

! RUN: bbc -fopenacc -emit-hlfir %s -o - | FileCheck %s

module acc_routines

! CHECK: acc.routine @acc_routine_1 func(@_QMacc_routinesPacc2)
! CHECK: acc.routine @acc_routine_0 func(@_QMacc_routinesPacc1) seq

!$acc routine(acc1) seq

contains

  subroutine acc1()
  end subroutine

! CHECK-LABEL: func.func @_QMacc_routinesPacc1() attributes {acc.routine_info = #acc.routine_info<[@acc_routine_0]>}

  subroutine acc2()
    !$acc routine(acc2)
  end subroutine

! CHECK-LABEL: func.func @_QMacc_routinesPacc2() attributes {acc.routine_info = #acc.routine_info<[@acc_routine_1]>}

end module

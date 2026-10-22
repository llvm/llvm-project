! This test checks lowering of OpenACC routine directive.

! RUN: bbc -fopenacc -emit-hlfir %s -o - | FileCheck %s

module acc_routines

! CHECK: acc.routine @[[r0:.*]] func(@_QMacc_routinesPacc2)
! CHECK: acc.routine @[[r1:.*]] func(@_QMacc_routinesPacc1) seq

!$acc routine(acc1) seq

contains

  subroutine acc1()
  end subroutine

! CHECK-LABEL: func.func @_QMacc_routinesPacc1()
! CHECK-SAME:attributes {acc.routine_info = #acc.routine_info<[@[[r1]]]>}

  subroutine acc2()
    !$acc routine(acc2)
  end subroutine

! CHECK-LABEL: func.func @_QMacc_routinesPacc2()
! CHECK-SAME:attributes {acc.routine_info = #acc.routine_info<[@[[r0]]]>}

end module

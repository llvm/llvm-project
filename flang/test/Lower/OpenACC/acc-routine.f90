! This test checks lowering of OpenACC routine directive.

! RUN: bbc -fopenacc -emit-fir %s -o - | FileCheck %s

! CHECK: acc.routine @acc_routine_6 func(@_QPacc_routine7) gang(dim = 1 : i32)
! CHECK: acc.routine @acc_routine_5 func(@_QPacc_routine6) nohost
! CHECK: acc.routine @acc_routine_4 func(@_QPacc_routine5) worker
! CHECK: acc.routine @acc_routine_3 func(@_QPacc_routine4) vector
! CHECK: acc.routine @acc_routine_2 func(@_QPacc_routine3) gang
! CHECK: acc.routine @acc_routine_1 func(@_QPacc_routine2) seq
! CHECK: acc.routine @acc_routine_0 func(@_QPacc_routine1)

subroutine acc_routine1()
  !$acc routine
end subroutine

! CHECK-LABEL: func.func @_QPacc_routine1() attributes {acc.routine_info = #acc.routine_info<[@acc_routine_0]>}

subroutine acc_routine2()
  !$acc routine seq
end subroutine

! CHECK-LABEL: func.func @_QPacc_routine2() attributes {acc.routine_info = #acc.routine_info<[@acc_routine_1]>}

subroutine acc_routine3()
  !$acc routine gang
end subroutine

! CHECK-LABEL: func.func @_QPacc_routine3() attributes {acc.routine_info = #acc.routine_info<[@acc_routine_2]>}

subroutine acc_routine4()
  !$acc routine vector
end subroutine

! CHECK-LABEL: func.func @_QPacc_routine4() attributes {acc.routine_info = #acc.routine_info<[@acc_routine_3]>}

subroutine acc_routine5()
  !$acc routine worker
end subroutine

! CHECK-LABEL: func.func @_QPacc_routine5() attributes {acc.routine_info = #acc.routine_info<[@acc_routine_4]>}

subroutine acc_routine6()
  !$acc routine nohost
end subroutine

! CHECK-LABEL: func.func @_QPacc_routine6() attributes {acc.routine_info = #acc.routine_info<[@acc_routine_5]>}

subroutine acc_routine7()
  !$acc routine gang(dim:1)
end subroutine

! CHECK-LABEL: func.func @_QPacc_routine7() attributes {acc.routine_info = #acc.routine_info<[@acc_routine_6]>}

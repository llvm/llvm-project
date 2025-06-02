! This test checks lowering of OpenACC routine directive.

! RUN: bbc -fopenacc -emit-hlfir %s -o - | FileCheck %s

! CHECK: acc.routine @[[r14:.*]] func(@_QPacc_routine19) bind("_QPacc_routine17" [#acc.device_type<host>], "_QPacc_routine17" [#acc.device_type<default>], "_QPacc_routine16" [#acc.device_type<multicore>])
! CHECK: acc.routine @[[r13:.*]] func(@_QPacc_routine18) bind("_QPacc_routine17" [#acc.device_type<host>], "_QPacc_routine16" [#acc.device_type<multicore>])
! CHECK: acc.routine @[[r12:.*]] func(@_QPacc_routine17) worker ([#acc.device_type<host>]) vector ([#acc.device_type<multicore>])
! CHECK: acc.routine @[[r11:.*]] func(@_QPacc_routine16) gang([#acc.device_type<nvidia>]) seq ([#acc.device_type<host>])
! CHECK: acc.routine @[[r10:.*]] func(@_QPacc_routine11) seq
! CHECK: acc.routine @[[r09:.*]] func(@_QPacc_routine10) seq
! CHECK: acc.routine @[[r08:.*]] func(@_QPacc_routine9) bind("_QPacc_routine9a")
! CHECK: acc.routine @[[r07:.*]] func(@_QPacc_routine8) bind("routine8_")
! CHECK: acc.routine @[[r06:.*]] func(@_QPacc_routine7) gang(dim: 1 : i64)
! CHECK: acc.routine @[[r05:.*]] func(@_QPacc_routine6) nohost
! CHECK: acc.routine @[[r04:.*]] func(@_QPacc_routine5) worker
! CHECK: acc.routine @[[r03:.*]] func(@_QPacc_routine4) vector
! CHECK: acc.routine @[[r02:.*]] func(@_QPacc_routine3) gang
! CHECK: acc.routine @[[r01:.*]] func(@_QPacc_routine2) seq
! CHECK: acc.routine @[[r00:.*]] func(@_QPacc_routine1)

subroutine acc_routine1()
  !$acc routine
end subroutine

! CHECK-LABEL: func.func @_QPacc_routine1()
! CHECK-SAME: attributes {acc.routine_info = #acc.routine_info<[@[[r00]]]>}

subroutine acc_routine2()
  !$acc routine seq
end subroutine

! CHECK-LABEL: func.func @_QPacc_routine2()
! CHECK-SAME: attributes {acc.routine_info = #acc.routine_info<[@[[r01]]]>}

subroutine acc_routine3()
  !$acc routine gang
end subroutine

! CHECK-LABEL: func.func @_QPacc_routine3()
! CHECK-SAME: attributes {acc.routine_info = #acc.routine_info<[@[[r02]]]>}

subroutine acc_routine4()
  !$acc routine vector
end subroutine

! CHECK-LABEL: func.func @_QPacc_routine4()
! CHECK-SAME: attributes {acc.routine_info = #acc.routine_info<[@[[r03]]]>}

subroutine acc_routine5()
  !$acc routine worker
end subroutine

! CHECK-LABEL: func.func @_QPacc_routine5()
! CHECK-SAME: attributes {acc.routine_info = #acc.routine_info<[@[[r04]]]>}

subroutine acc_routine6()
  !$acc routine nohost
end subroutine

! CHECK-LABEL: func.func @_QPacc_routine6()
! CHECK-SAME: attributes {acc.routine_info = #acc.routine_info<[@[[r05]]]>}

subroutine acc_routine7()
  !$acc routine gang(dim:1)
end subroutine

! CHECK-LABEL: func.func @_QPacc_routine7()
! CHECK-SAME: attributes {acc.routine_info = #acc.routine_info<[@[[r06]]]>}

subroutine acc_routine8()
  !$acc routine bind("routine8_")
end subroutine

! CHECK-LABEL: func.func @_QPacc_routine8()
! CHECK-SAME: attributes {acc.routine_info = #acc.routine_info<[@[[r07]]]>}

subroutine acc_routine9a()
end subroutine

subroutine acc_routine9()
  !$acc routine bind(acc_routine9a)
end subroutine

! CHECK-LABEL: func.func @_QPacc_routine9()
! CHECK-SAME: attributes {acc.routine_info = #acc.routine_info<[@[[r08]]]>}

function acc_routine10()
  !$acc routine(acc_routine10) seq
end function

! CHECK-LABEL: func.func @_QPacc_routine10() -> f32
! CHECK-SAME: attributes {acc.routine_info = #acc.routine_info<[@[[r09]]]>}

subroutine acc_routine11(a)
  real :: a
  !$acc routine(acc_routine11) seq
end subroutine

! CHECK-LABEL: func.func @_QPacc_routine11(%arg0: !fir.ref<f32> {fir.bindc_name = "a"})
! CHECK-SAME: attributes {acc.routine_info = #acc.routine_info<[@[[r10]]]>}

subroutine acc_routine12()

  interface
  subroutine acc_routine11(a)
    real :: a
    !$acc routine(acc_routine11) seq
  end subroutine
  end interface

end subroutine

subroutine acc_routine13()
  !$acc routine bind(acc_routine14)
end subroutine

subroutine acc_routine14()
end subroutine

subroutine acc_routine15()
  !$acc routine bind(acc_routine16)
end subroutine

subroutine acc_routine16()
  !$acc routine device_type(host) seq dtype(nvidia) gang
end subroutine

subroutine acc_routine17()
  !$acc routine device_type(host) worker dtype(multicore) vector 
end subroutine

subroutine acc_routine18()
  !$acc routine device_type(host) bind(acc_routine17) dtype(multicore) bind(acc_routine16) 
end subroutine

subroutine acc_routine19()
  !$acc routine device_type(host,default) bind(acc_routine17) dtype(multicore) bind(acc_routine16) 
end subroutine

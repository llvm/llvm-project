! This test checks lowering of OpenACC routine directive in interfaces.

! RUN: bbc -fopenacc -emit-hlfir %s -o - | FileCheck %s


subroutine sub1(a)
  !$acc routine worker bind(sub2)
  real :: a(:)
end subroutine

subroutine sub2(a)
  !$acc routine worker nohost
  real :: a(:)
end subroutine

subroutine test

interface
  subroutine sub1(a)
    !$acc routine worker bind(sub2)
    real :: a(:)
  end subroutine
 
  subroutine sub2(a)
    !$acc routine worker nohost
    real :: a(:)
  end subroutine
end interface

end subroutine

! CHECK: acc.routine @acc_routine_1 func(@_QPsub2) worker nohost
! CHECK: acc.routine @acc_routine_0 func(@_QPsub1) bind("_QPsub2") worker
! CHECK: func.func @_QPsub1(%arg0: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "a"}) attributes {acc.routine_info = #acc.routine_info<[@acc_routine_0]>}
! CHECK: func.func @_QPsub2(%arg0: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "a"}) attributes {acc.routine_info = #acc.routine_info<[@acc_routine_1]>}

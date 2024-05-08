! This test checks correct lowering when OpenACC routine directive is placed
! before implicit none.

! RUN: bbc -fopenacc -emit-hlfir %s -o - | FileCheck %s

module dummy_mod
contains

  subroutine sub1(i)
    !$acc routine seq
    integer :: i
  end subroutine
end module

program test_acc_routine
  use dummy_mod
  
  !$acc routine(sub2) seq
  
  implicit none
  
  integer :: i

contains
  subroutine sub2()
  end subroutine
  
end program

! CHECK: acc.routine @acc_routine_1 func(@_QFPsub2) seq
! CHECK: acc.routine @acc_routine_0 func(@_QMdummy_modPsub1) seq
! CHECK: func.func @_QMdummy_modPsub1(%arg0: !fir.ref<i32> {fir.bindc_name = "i"}) attributes {acc.routine_info = #acc.routine_info<[@acc_routine_0]>}
! CHECK: func.func @_QQmain() attributes {fir.bindc_name = "test_acc_routine"}
! CHECK: func.func private @_QFPsub2() attributes {acc.routine_info = #acc.routine_info<[@acc_routine_1]>, fir.host_symbol = @_QQmain, llvm.linkage = #llvm.linkage<internal>}

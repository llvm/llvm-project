! RUN: bbc -emit-fir -o - %s | FileCheck %s --implicit-check-not=fir.if --implicit-check-not="fir.call @_QPdead"
! CHECK-LABEL: func.func @_QPconstant_branches(
! CHECK: fir.call @_QPlive_true()
! CHECK-NEXT: fir.call @_QPlive_false()
! CHECK-NEXT: return
subroutine constant_branches()
  logical, parameter :: enabled = .false.
  if (.true.) then
    call live_true()
  else
    call dead()
  end if
  if (enabled) then
    call dead()
  else
    call live_false()
  end if
end subroutine
! CHECK-LABEL: func.func @_QPelse_if_chain(
! CHECK: fir.call @_QPlive_chain()
! CHECK-NEXT: return
subroutine else_if_chain(c)
  logical :: c
  if (.false.) then
    call dead()
  else if (.true.) then
    call live_chain()
  else if (c) then
    call dead()
  else
    call dead()
  end if
end subroutine

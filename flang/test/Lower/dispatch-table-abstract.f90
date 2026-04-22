! Test lowering of ASBTRACT type to fir.type_info
! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

module m_abstract_info
  type, abstract :: abstract_type
    contains
    procedure(proc_iface), nopass, deferred :: proc
  end type
  interface
    subroutine proc_iface()
    end subroutine
  end interface
end module

subroutine test(x)
  use m_abstract_info, only : abstract_type
  class(abstract_type) :: x
end subroutine

!CHECK-LABEL:  fir.type_info @_QMm_abstract_infoTabstract_type abstract noinit nodestroy nofinal : !fir.type<_QMm_abstract_infoTabstract_type> dispatch_table {
!CHECK:    fir.dt_entry "proc", @_QPproc_iface deferred

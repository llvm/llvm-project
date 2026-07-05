! Test instantiation of procedure pointer hidden behind
! generic interface host associated from a module.

! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

module m_proc_pointer_hidden_by_generic
procedure (proc2),pointer :: p
interface p
  procedure proc1, p
end interface
contains
subroutine proc1(i)
  integer :: i
end subroutine
subroutine proc2(x)
  real :: x
end subroutine

subroutine test()
p=>proc2
end subroutine
end

! CHECK-LABEL:   func.func @_QMm_proc_pointer_hidden_by_genericPtest(
! CHECK:           %[[DUMMY_SCOPE_0:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[ADDRESS_OF_0:.*]] = fir.address_of(@_QMm_proc_pointer_hidden_by_genericEp) : !fir.ref<!fir.boxproc<(!fir.ref<f32>) -> ()>>
! CHECK:           %[[DECLARE_0:.*]]:2 = hlfir.declare %[[ADDRESS_OF_0]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QMm_proc_pointer_hidden_by_genericEp"} : (!fir.ref<!fir.boxproc<(!fir.ref<f32>) -> ()>>) -> (!fir.ref<!fir.boxproc<(!fir.ref<f32>) -> ()>>, !fir.ref<!fir.boxproc<(!fir.ref<f32>) -> ()>>)
! CHECK:           %[[ADDRESS_OF_1:.*]] = fir.address_of(@_QMm_proc_pointer_hidden_by_genericPproc2) : (!fir.ref<f32>) -> ()
! CHECK:           %[[EMBOXPROC_0:.*]] = fir.emboxproc %[[ADDRESS_OF_1]] : ((!fir.ref<f32>) -> ()) -> !fir.boxproc<() -> ()>
! CHECK:           %[[CONVERT_0:.*]] = fir.convert %[[EMBOXPROC_0]] : (!fir.boxproc<() -> ()>) -> !fir.boxproc<(!fir.ref<f32>) -> ()>
! CHECK:           fir.store %[[CONVERT_0]] to %[[DECLARE_0]]#0 : !fir.ref<!fir.boxproc<(!fir.ref<f32>) -> ()>>
! CHECK:           return
! CHECK:         }

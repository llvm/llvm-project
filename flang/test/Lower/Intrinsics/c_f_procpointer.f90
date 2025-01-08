! Test C_F_PROCPOINTER() lowering.
! RUN: bbc -emit-hlfir -o - %s | FileCheck %s

subroutine test_c_funloc(fptr, cptr)
  use iso_c_binding, only : c_f_procpointer, c_funptr
  real, pointer, external :: fptr
  type(c_funptr), cptr
  call c_f_procpointer(cptr, fptr)
end subroutine
! CHECK-LABEL:   func.func @_QPtest_c_funloc(
! CHECK-SAME:                                %[[VAL_0:.*]]: !fir.ref<!fir.boxproc<() -> ()>>,
! CHECK-SAME:                                %[[VAL_1:.*]]: !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>> {fir.bindc_name = "cptr"}) {
! CHECK:           %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_1]] dummy_scope %{{[0-9]+}} {uniq_name = "_QFtest_c_funlocEcptr"} : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>>, !fir.dscope) -> (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>>, !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>>)
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %{{[0-9]+}} {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFtest_c_funlocEfptr"} : (!fir.ref<!fir.boxproc<() -> ()>>, !fir.dscope) -> (!fir.ref<!fir.boxproc<() -> ()>>, !fir.ref<!fir.boxproc<() -> ()>>)
! CHECK:           %[[VAL_4:.*]] = fir.field_index __address, !fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>
! CHECK:           %[[VAL_5:.*]] = fir.coordinate_of %[[VAL_2]]#1, %[[VAL_4]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>>, !fir.field) -> !fir.ref<i64>
! CHECK:           %[[VAL_6:.*]] = fir.load %[[VAL_5]] : !fir.ref<i64>
! CHECK:           %[[VAL_7:.*]] = fir.convert %[[VAL_6]] : (i64) -> (() -> ())
! CHECK:           %[[VAL_8:.*]] = fir.emboxproc %[[VAL_7]] : (() -> ()) -> !fir.boxproc<() -> ()>
! CHECK:           fir.store %[[VAL_8]] to %[[VAL_3]]#1 : !fir.ref<!fir.boxproc<() -> ()>>

subroutine test_c_funloc_char(fptr, cptr)
  use iso_c_binding, only : c_f_procpointer, c_funptr
  interface
    character(10) function char_func()
    end function
  end interface
  procedure(char_func), pointer :: fptr
  type(c_funptr), cptr
  call c_f_procpointer(cptr, fptr)
end subroutine
! CHECK-LABEL:   func.func @_QPtest_c_funloc_char(
! CHECK-SAME:                                     %[[VAL_0:.*]]: !fir.ref<!fir.boxproc<() -> ()>>,
! CHECK-SAME:                                     %[[VAL_1:.*]]: !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>> {fir.bindc_name = "cptr"}) {
! CHECK:           %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_1]] dummy_scope %{{[0-9]+}} {uniq_name = "_QFtest_c_funloc_charEcptr"} : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>>, !fir.dscope) -> (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>>, !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>>)
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %{{[0-9]+}} {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFtest_c_funloc_charEfptr"} : (!fir.ref<!fir.boxproc<() -> ()>>, !fir.dscope) -> (!fir.ref<!fir.boxproc<() -> ()>>, !fir.ref<!fir.boxproc<() -> ()>>)
! CHECK:           %[[VAL_4:.*]] = fir.field_index __address, !fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>
! CHECK:           %[[VAL_5:.*]] = fir.coordinate_of %[[VAL_2]]#1, %[[VAL_4]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>>, !fir.field) -> !fir.ref<i64>
! CHECK:           %[[VAL_6:.*]] = fir.load %[[VAL_5]] : !fir.ref<i64>
! CHECK:           %[[VAL_7:.*]] = fir.convert %[[VAL_6]] : (i64) -> (() -> ())
! CHECK:           %[[VAL_8:.*]] = fir.emboxproc %[[VAL_7]] : (() -> ()) -> !fir.boxproc<() -> ()>
! CHECK:           fir.store %[[VAL_8]] to %[[VAL_3]]#1 : !fir.ref<!fir.boxproc<() -> ()>>

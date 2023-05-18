! Test C_FUNLOC() with procedure pointers.
! RUN: bbc -emit-hlfir -o - %s | FileCheck %s

subroutine test_c_funloc(p)
  use iso_c_binding, only : c_funloc
  real, pointer, external :: p
  call test(c_funloc(p))
end subroutine
! CHECK-LABEL:   func.func @_QPtest_c_funloc(
! CHECK-SAME:                                %[[VAL_0:.*]]: !fir.ref<!fir.boxproc<() -> ()>>) {
! CHECK:           %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFtest_c_funlocEp"} : (!fir.ref<!fir.boxproc<() -> ()>>) -> (!fir.ref<!fir.boxproc<() -> ()>>, !fir.ref<!fir.boxproc<() -> ()>>)
! CHECK:           %[[VAL_2:.*]] = fir.load %[[VAL_1]]#0 : !fir.ref<!fir.boxproc<() -> ()>>
! CHECK:           %[[VAL_3:.*]] = fir.alloca !fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>
! CHECK:           %[[VAL_4:.*]] = fir.field_index __address, !fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>
! CHECK:           %[[VAL_5:.*]] = fir.coordinate_of %[[VAL_3]], %[[VAL_4]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>>, !fir.field) -> !fir.ref<i64>
! CHECK:           %[[VAL_6:.*]] = fir.box_addr %[[VAL_2]] : (!fir.boxproc<() -> ()>) -> (() -> ())
! CHECK:           %[[VAL_7:.*]] = fir.convert %[[VAL_6]] : (() -> ()) -> i64
! CHECK:           fir.store %[[VAL_7]] to %[[VAL_5]] : !fir.ref<i64>

subroutine test_c_funloc_char(p)
  use iso_c_binding, only : c_funloc
  interface
    character(10) function char_func()
    end function
  end interface
  procedure(char_func), pointer :: p
  call test(c_funloc(p))
end subroutine
! CHECK-LABEL:   func.func @_QPtest_c_funloc_char(
! CHECK-SAME:                                     %[[VAL_0:.*]]: !fir.ref<!fir.boxproc<() -> ()>>) {
! CHECK:           %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFtest_c_funloc_charEp"} : (!fir.ref<!fir.boxproc<() -> ()>>) -> (!fir.ref<!fir.boxproc<() -> ()>>, !fir.ref<!fir.boxproc<() -> ()>>)
! CHECK:           %[[VAL_2:.*]] = fir.load %[[VAL_1]]#0 : !fir.ref<!fir.boxproc<() -> ()>>
! CHECK:           %[[VAL_3:.*]] = fir.alloca !fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>
! CHECK:           %[[VAL_4:.*]] = fir.field_index __address, !fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>
! CHECK:           %[[VAL_5:.*]] = fir.coordinate_of %[[VAL_3]], %[[VAL_4]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>>, !fir.field) -> !fir.ref<i64>
! CHECK:           %[[VAL_6:.*]] = fir.box_addr %[[VAL_2]] : (!fir.boxproc<() -> ()>) -> (() -> ())
! CHECK:           %[[VAL_7:.*]] = fir.convert %[[VAL_6]] : (() -> ()) -> i64
! CHECK:           fir.store %[[VAL_7]] to %[[VAL_5]] : !fir.ref<i64>

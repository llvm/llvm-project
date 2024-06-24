! RUN: bbc -emit-hlfir %s -o - | FileCheck %s

! CHECK-LABEL:   func.func @_QPtest1() {
! CHECK:           %[[VAL_110:.*]]:3 = hlfir.associate %{{.*}} {uniq_name = "adapt.cptrbyval"} : (!hlfir.expr<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>) -> (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>, !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>, i1)
! CHECK:           %[[VAL_111:.*]] = fir.field_index __address, !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
! CHECK:           %[[VAL_112:.*]] = fir.coordinate_of %[[VAL_110]]#1, %[[VAL_111]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>, !fir.field) -> !fir.ref<i64>
! CHECK:           %[[VAL_113:.*]] = fir.load %[[VAL_112]] : !fir.ref<i64>
! CHECK:           %[[VAL_114:.*]] = fir.convert %[[VAL_113]] : (i64) -> !fir.ref<i64>
! CHECK:           hlfir.end_associate %[[VAL_110]]#1, %[[VAL_110]]#2 : !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>, i1
! CHECK:           fir.call @get_expected_f(%[[VAL_114]]) fastmath<contract> {is_bind_c} : (!fir.ref<i64>) -> ()
subroutine test1
  use iso_c_binding
  interface
     subroutine get_expected_f(src) bind(c)
       use iso_c_binding
       type(c_ptr), value :: src
     end subroutine get_expected_f
  end interface
  real, target,  dimension(1) :: r_src
  call get_expected_f(c_loc(r_src))
end

! CHECK-LABEL:   func.func @_QPtest2(
! CHECK-SAME:                        %[[VAL_0:.*]]: !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>> {fir.bindc_name = "cptr"}) {
! CHECK:           %[[DSCOPE:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_97:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %[[DSCOPE]] {uniq_name = "_QFtest2Ecptr"} : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>, !fir.dscope) -> (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>, !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>)
! CHECK:           %[[VAL_98:.*]] = fir.field_index __address, !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
! CHECK:           %[[VAL_99:.*]] = fir.coordinate_of %[[VAL_97]]#0, %[[VAL_98]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>, !fir.field) -> !fir.ref<i64>
! CHECK:           %[[VAL_100:.*]] = fir.load %[[VAL_99]] : !fir.ref<i64>
! CHECK:           %[[VAL_101:.*]] = fir.convert %[[VAL_100]] : (i64) -> !fir.ref<i64>
! CHECK:           fir.call @get_expected_f(%[[VAL_101]]) fastmath<contract> {is_bind_c} : (!fir.ref<i64>) -> ()
subroutine test2(cptr)
  use iso_c_binding
  interface
     subroutine get_expected_f(src) bind(c)
       use iso_c_binding
       type(c_ptr), value :: src
     end subroutine get_expected_f
  end interface
  type(c_ptr) :: cptr
  call get_expected_f(cptr)
end

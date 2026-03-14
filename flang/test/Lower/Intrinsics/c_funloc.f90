! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

! Test intrinsic module procedure c_funloc

! CHECK-LABEL: func.func @_QPtest() {
! CHECK-DAG:     %[[TMP_CPTR_ALLOCA:.*]] = fir.alloca !fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}> {bindc_name = "tmp_cptr", uniq_name = "_QFtestEtmp_cptr"}
! CHECK-DAG:     %[[TMP_CPTR:.*]]:2 = hlfir.declare %[[TMP_CPTR_ALLOCA]] {uniq_name = "_QFtestEtmp_cptr"}
! CHECK:         %[[VAL_1:.*]] = fir.address_of(@_QPfoo) : (!fir.ref<i32>) -> ()
! CHECK:         %[[VAL_2:.*]] = fir.emboxproc %[[VAL_1]] : ((!fir.ref<i32>) -> ()) -> !fir.boxproc<() -> ()>
! CHECK:         %[[RESULT_ALLOCA:.*]] = fir.alloca !fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>
! CHECK:         %[[VAL_7:.*]] = fir.coordinate_of %[[RESULT_ALLOCA]], __address : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>>) -> !fir.ref<i64>
! CHECK:         %[[VAL_4:.*]] = fir.box_addr %[[VAL_2]] : (!fir.boxproc<() -> ()>) -> (() -> ())
! CHECK:         %[[VAL_5:.*]] = fir.convert %[[VAL_4]] : (() -> ()) -> i64
! CHECK:         fir.store %[[VAL_5]] to %[[VAL_7]] : !fir.ref<i64>
! CHECK:         %[[RESULT_DECL:.*]]:2 = hlfir.declare %[[RESULT_ALLOCA]] {uniq_name = ".tmp.intrinsic_result"}
! CHECK:         %[[RESULT_EXPR:.*]] = hlfir.as_expr %[[RESULT_DECL]]#0 move %{{.*}} : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>>, i1) -> !hlfir.expr<!fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>>
! CHECK:         hlfir.assign %[[RESULT_EXPR]] to %[[TMP_CPTR]]#0 : !hlfir.expr<!fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>>, !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>>
! CHECK:         hlfir.destroy %[[RESULT_EXPR]] : !hlfir.expr<!fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>>

subroutine test()
  use iso_c_binding
  interface
    subroutine foo(i)
      integer :: i
    end
  end interface

  type(c_funptr) :: tmp_cptr

  tmp_cptr = c_funloc(foo)
end

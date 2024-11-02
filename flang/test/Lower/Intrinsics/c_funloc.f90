! RUN: bbc -emit-fir %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-fir %s -o - | FileCheck %s

! Test intrinsic module procedure c_funloc

! CHECK-LABEL: func.func @_QPtest() {
! CHECK:         %[[VAL_1:.*]] = fir.address_of(@_QPfoo) : (!fir.ref<i32>) -> ()
! CHECK:         %[[VAL_2:.*]] = fir.emboxproc %[[VAL_1]] : ((!fir.ref<i32>) -> ()) -> !fir.boxproc<(!fir.ref<i32>) -> ()>
! CHECK:         %[[VAL_3:.*]] = fir.alloca !fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>
! CHECK-DAG:         %[[VAL_4:.*]] = fir.box_addr %[[VAL_2]] : (!fir.boxproc<(!fir.ref<i32>) -> ()>) -> ((!fir.ref<i32>) -> ())
! CHECK-DAG:         %[[VAL_5:.*]] = fir.convert %[[VAL_4]] : ((!fir.ref<i32>) -> ()) -> i64
! CHECK-DAG:         %[[VAL_6:.*]] = fir.field_index __address, !fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>
! CHECK-DAG:         %[[VAL_7:.*]] = fir.coordinate_of %[[VAL_3]], %[[VAL_6]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>>, !fir.field) -> !fir.ref<i64>
! CHECK:         fir.store %[[VAL_5]] to %[[VAL_7]] : !fir.ref<i64>

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

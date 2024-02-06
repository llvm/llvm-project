! Test lowering of C_NULL_PTR in structure constructor initial value.
! RUN: bbc -emit-hlfir -o - %s | FileCheck %s
subroutine test
  use, intrinsic :: iso_c_binding, only : c_ptr, c_null_ptr
  type t
     type(c_ptr) :: ptr
  end type t
  type(t) :: x = t(c_null_ptr)
end subroutine
! CHECK-LABEL:  fir.global internal @_QFtestEx : !fir.type<_QFtestTt{ptr:!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>}> {
! CHECK:           %[[VAL_0:.*]] = fir.undefined !fir.type<_QFtestTt{ptr:!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>}>
! CHECK:           %[[VAL_1:.*]] = fir.field_index ptr, !fir.type<_QFtestTt{ptr:!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>}>
! CHECK:           %[[VAL_2:.*]] = fir.undefined !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
! CHECK:           %[[VAL_3:.*]] = fir.field_index __address, !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
! CHECK:           %[[VAL_4:.*]] = arith.constant 0 : i64
! CHECK:           %[[VAL_5:.*]] = fir.insert_value %[[VAL_2]], %[[VAL_4]], ["__address", !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>] : (!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>, i64) -> !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
! CHECK:           %[[VAL_6:.*]] = fir.insert_value %[[VAL_0]], %[[VAL_5]], ["ptr", !fir.type<_QFtestTt{ptr:!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>}>] : (!fir.type<_QFtestTt{ptr:!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>}>, !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>) -> !fir.type<_QFtestTt{ptr:!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>}>
! CHECK:           fir.has_value %[[VAL_6]] : !fir.type<_QFtestTt{ptr:!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>}>
! CHECK:         }

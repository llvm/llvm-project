! Test creation of outlined literal array with c_ptr/c_funptr elements.
! RUN: bbc -emit-hlfir -o - %s | FileCheck %s

subroutine test
  use, intrinsic :: iso_c_binding
  type t1
     type(c_ptr) :: d(1)
  end type t1
  type(t1), parameter :: x(1) = t1(c_null_ptr)
  type(t1) :: y(1)
  y = x(1)
end subroutine test
! CHECK-LABEL:   fir.global internal @_QQro._QFtestTt1.0 constant : !fir.type<_QFtestTt1{d:!fir.array<1x!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>}> {
! CHECK:           %[[VAL_0:.*]] = fir.undefined !fir.type<_QFtestTt1{d:!fir.array<1x!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>}>
! CHECK:           %[[VAL_1:.*]] = fir.field_index d, !fir.type<_QFtestTt1{d:!fir.array<1x!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>}>
! CHECK:           %[[VAL_2:.*]] = fir.undefined !fir.array<1x!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>
! CHECK:           %[[VAL_3:.*]] = fir.undefined !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
! CHECK:           %[[VAL_4:.*]] = fir.field_index __address, !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
! CHECK:           %[[VAL_5:.*]] = arith.constant 0 : i64
! CHECK:           %[[VAL_6:.*]] = fir.insert_value %[[VAL_3]], %[[VAL_5]], ["__address", !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>] : (!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>, i64) -> !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
! CHECK:           %[[VAL_7:.*]] = fir.insert_value %[[VAL_2]], %[[VAL_6]], [0 : index] : (!fir.array<1x!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>, !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>) -> !fir.array<1x!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>
! CHECK:           %[[VAL_8:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_9:.*]] = fir.insert_value %[[VAL_0]], %[[VAL_7]], ["d", !fir.type<_QFtestTt1{d:!fir.array<1x!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>}>] : (!fir.type<_QFtestTt1{d:!fir.array<1x!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>}>, !fir.array<1x!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>) -> !fir.type<_QFtestTt1{d:!fir.array<1x!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>}>
! CHECK:           fir.has_value %[[VAL_9]] : !fir.type<_QFtestTt1{d:!fir.array<1x!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>}>
! CHECK:         }


subroutine test2
  use, intrinsic :: iso_c_binding
  type t1
     type(c_funptr) :: d(1)
  end type t1
  type(t1), parameter :: x(1) = t1(c_null_funptr)
  type(t1) :: y(1)
  y = x(1)
end subroutine test2
! CHECK-LABEL:   fir.global internal @_QQro._QFtest2Tt1.1 constant : !fir.type<_QFtest2Tt1{d:!fir.array<1x!fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>>}> {
! CHECK:           %[[VAL_0:.*]] = fir.undefined !fir.type<_QFtest2Tt1{d:!fir.array<1x!fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>>}>
! CHECK:           %[[VAL_1:.*]] = fir.field_index d, !fir.type<_QFtest2Tt1{d:!fir.array<1x!fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>>}>
! CHECK:           %[[VAL_2:.*]] = fir.undefined !fir.array<1x!fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>>
! CHECK:           %[[VAL_3:.*]] = fir.undefined !fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>
! CHECK:           %[[VAL_4:.*]] = fir.field_index __address, !fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>
! CHECK:           %[[VAL_5:.*]] = arith.constant 0 : i64
! CHECK:           %[[VAL_6:.*]] = fir.insert_value %[[VAL_3]], %[[VAL_5]], ["__address", !fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>] : (!fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>, i64) -> !fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>
! CHECK:           %[[VAL_7:.*]] = fir.insert_value %[[VAL_2]], %[[VAL_6]], [0 : index] : (!fir.array<1x!fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>>, !fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>) -> !fir.array<1x!fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>>
! CHECK:           %[[VAL_8:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_9:.*]] = fir.insert_value %[[VAL_0]], %[[VAL_7]], ["d", !fir.type<_QFtest2Tt1{d:!fir.array<1x!fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>>}>] : (!fir.type<_QFtest2Tt1{d:!fir.array<1x!fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>>}>, !fir.array<1x!fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>>) -> !fir.type<_QFtest2Tt1{d:!fir.array<1x!fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>>}>
! CHECK:           fir.has_value %[[VAL_9]] : !fir.type<_QFtest2Tt1{d:!fir.array<1x!fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>>}>
! CHECK:         }

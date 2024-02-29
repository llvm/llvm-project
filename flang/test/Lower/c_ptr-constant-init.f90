! Test creation of outlined literal array with c_ptr/c_funptr elements.
! RUN: bbc -emit-fir -hlfir=false -o - %s | FileCheck %s

subroutine test
  use, intrinsic :: iso_c_binding
  type t1
     type(c_ptr) :: d(1)
  end type t1
  type(t1), parameter :: x(1) = t1(c_null_ptr)
  type(t1) :: y(1)
  y = x(1)
end subroutine test
! CHECK-LABEL:   fir.global internal @_QQro.1x_QM__fortran_builtinsT__builtin_c_ptr.0 constant : !fir.array<1x!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>> {
! CHECK:           %[[VAL_0:.*]] = fir.undefined !fir.array<1x!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>
! CHECK:           %[[VAL_1:.*]] = fir.undefined !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
! CHECK:           %[[VAL_2:.*]] = fir.field_index __address, !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
! CHECK:           %[[VAL_3:.*]] = arith.constant 0 : i64
! CHECK:           %[[VAL_4:.*]] = fir.insert_value %[[VAL_1]], %[[VAL_3]], ["__address", !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>] : (!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>, i64) -> !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
! CHECK:           %[[VAL_5:.*]] = fir.insert_value %[[VAL_0]], %[[VAL_4]], [0 : index] : (!fir.array<1x!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>, !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>) -> !fir.array<1x!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>
! CHECK:           fir.has_value %[[VAL_5]] : !fir.array<1x!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>
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
! CHECK-LABEL:   fir.global internal @_QQro.1x_QM__fortran_builtinsT__builtin_c_funptr.1 constant : !fir.array<1x!fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>> {
! CHECK:           %[[VAL_0:.*]] = fir.undefined !fir.array<1x!fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>>
! CHECK:           %[[VAL_1:.*]] = fir.undefined !fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>
! CHECK:           %[[VAL_2:.*]] = fir.field_index __address, !fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>
! CHECK:           %[[VAL_3:.*]] = arith.constant 0 : i64
! CHECK:           %[[VAL_4:.*]] = fir.insert_value %[[VAL_1]], %[[VAL_3]], ["__address", !fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>] : (!fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>, i64) -> !fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>
! CHECK:           %[[VAL_5:.*]] = fir.insert_value %[[VAL_0]], %[[VAL_4]], [0 : index] : (!fir.array<1x!fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>>, !fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>) -> !fir.array<1x!fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>>
! CHECK:           fir.has_value %[[VAL_5]] : !fir.array<1x!fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>>
! CHECK:         }

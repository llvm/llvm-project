! Test creation of outlined literal array with c_ptr/c_funptr elements.
! RUN: %flang_fc1 -emit-hlfir -o - %s | FileCheck %s

subroutine test
  use, intrinsic :: iso_c_binding
  type t1
     type(c_ptr) :: d(1)
  end type t1
  type(t1), parameter :: x(1) = t1(c_null_ptr)
  type(t1) :: y(1)
  y = x(1)
end subroutine test
! CHECK-LABEL: fir.global internal @_QQro._QFtestTt1.0 constant : !fir.type<_QFtestTt1{d:!fir.array<1x!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>}> {
! CHECK: %[[UNDEF:.*]] = fir.undefined !fir.type<_QFtestTt1{d:!fir.array<1x!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>}>
! CHECK: %[[D_UNDEF:.*]] = fir.undefined !fir.array<1x!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>
! CHECK: %[[CPTR_UNDEF:.*]] = fir.undefined !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
! CHECK: %[[ADDR:.*]] = fir.insert_value %[[CPTR_UNDEF]], %c0{{.*}}, ["__address", !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>]
! CHECK: %[[D_VAL:.*]] = fir.insert_value %[[D_UNDEF]], %[[ADDR]], [0 : index]
! CHECK: %[[RES:.*]] = fir.insert_value %[[UNDEF]], %[[D_VAL]], ["d", !fir.type<_QFtestTt1{d:!fir.array<1x!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>}>]
! CHECK: fir.has_value %[[RES]]
! CHECK: }

subroutine test2
  use, intrinsic :: iso_c_binding
  type t1
     type(c_funptr) :: d(1)
  end type t1
  type(t1), parameter :: x(1) = t1(c_null_funptr)
  type(t1) :: y(1)
  y = x(1)
end subroutine test2
! CHECK-LABEL: fir.global internal @_QQro._QFtest2Tt1.1 constant : !fir.type<_QFtest2Tt1{d:!fir.array<1x!fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>>}> {
! CHECK: %[[UNDEF:.*]] = fir.undefined !fir.type<_QFtest2Tt1{d:!fir.array<1x!fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>>}>
! CHECK: %[[D_UNDEF:.*]] = fir.undefined !fir.array<1x!fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>>
! CHECK: %[[CPTR_UNDEF:.*]] = fir.undefined !fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>
! CHECK: %[[ADDR:.*]] = fir.insert_value %[[CPTR_UNDEF]], %c0{{.*}}, ["__address", !fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>]
! CHECK: %[[D_VAL:.*]] = fir.insert_value %[[D_UNDEF]], %[[ADDR]], [0 : index]
! CHECK: %[[RES:.*]] = fir.insert_value %[[UNDEF]], %[[D_VAL]], ["d", !fir.type<_QFtest2Tt1{d:!fir.array<1x!fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>>}>]
! CHECK: fir.has_value %[[RES]]
! CHECK: }

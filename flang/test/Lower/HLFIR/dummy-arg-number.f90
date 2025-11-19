! Test that dummy argument positions are tracked in hlfir.declare
! RUN: bbc -emit-hlfir -o - %s | FileCheck %s

! CHECK-LABEL: func.func @_QPsingle_arg(
subroutine single_arg(n)
  integer :: n
  ! CHECK: hlfir.declare %{{.*}} dummy_scope %{{.*}} arg 1 {uniq_name = "_QFsingle_argEn"}
  print *, n
end subroutine

! CHECK-LABEL: func.func @_QPmultiple_args(
subroutine multiple_args(a, b, c)
  integer :: a, b, c
  ! CHECK-DAG: hlfir.declare %{{.*}} dummy_scope %{{.*}} arg 1 {uniq_name = "_QFmultiple_argsEa"}
  ! CHECK-DAG: hlfir.declare %{{.*}} dummy_scope %{{.*}} arg 2 {uniq_name = "_QFmultiple_argsEb"}
  ! CHECK-DAG: hlfir.declare %{{.*}} dummy_scope %{{.*}} arg 3 {uniq_name = "_QFmultiple_argsEc"}
  print *, a, b, c
end subroutine

! CHECK-LABEL: func.func @_QPchar_arg(
subroutine char_arg(str)
  character(len=5) :: str
  ! CHECK: hlfir.declare %{{.*}} typeparams %{{.*}} dummy_scope %{{.*}} arg 1 {uniq_name = "_QFchar_argEstr"}
  print *, str
end subroutine

! CHECK-LABEL: func.func @_QParray_arg(
subroutine array_arg(arr)
  integer :: arr(:)
  ! CHECK: hlfir.declare %{{.*}} dummy_scope %{{.*}} arg 1 {uniq_name = "_QFarray_argEarr"}
  print *, arr(1)
end subroutine

! Test that local variables do NOT get arg numbers
! CHECK-LABEL: func.func @_QPlocal_var()
subroutine local_var()
  integer :: x
  ! CHECK: hlfir.declare %{{[0-9]+}} {uniq_name = "_QFlocal_varEx"}
  x = 10
  print *, x
end subroutine

! Test mixed arguments and locals
! CHECK-LABEL: func.func @_QPmixed(
subroutine mixed(n)
  integer :: n
  integer :: local_x
  ! CHECK-DAG: hlfir.declare %{{[0-9]+}} {uniq_name = "_QFmixedElocal_x"}
  ! CHECK-DAG: hlfir.declare {{.*}} dummy_scope {{.*}} arg 1 {uniq_name = "_QFmixedEn"}
  local_x = n + 1
  print *, local_x
end subroutine


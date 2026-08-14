!RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 %s -o - | FileCheck %s
!RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 -fopenmp-is-target-device %s -o - | FileCheck %s

! Check that the INDIRECT clause on a DECLARE TARGET directive is lowered to the
! `indirect` field of the omp.declare_target attribute.

module functions
  implicit none
contains
  ! CHECK: func.func @_QMfunctionsPfunc_true({{.*}}attributes {omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (enter), indirect = true>}
  function func_true() result(i)
    !$omp declare target enter(func_true) indirect(.true.)
    character(1) :: i
    i = 'a'
  end function

  ! CHECK: func.func @_QMfunctionsPfunc_implicit({{.*}}attributes {omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (enter), indirect = true>}
  function func_implicit() result(i)
    !$omp declare target enter(func_implicit) indirect
    character(1) :: i
    i = 'b'
  end function

  ! A false value equals the attribute default, so no `indirect` field prints.
  ! CHECK: func.func @_QMfunctionsPfunc_false({{.*}}attributes {omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (enter)>}
  function func_false() result(i)
    !$omp declare target enter(func_false) indirect(.false.)
    character(1) :: i
    i = 'c'
  end function
end module
